"""
Activation steering via PyTorch forward hooks on transformer decoder layers.

Usage:
    steerer = ActivationSteerer(model, steering_vectors)
    steerer.register_hooks()                    # once, after model load

    # during inference
    with steerer.steering_context(scale=1.0):
        output = model.generate(...)

    steerer.remove_hooks()                      # cleanup
"""
import logging
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Layers to attach steering hooks (layers 8–20 inclusive, matching Task 2 analysis)
STEER_LAYERS: List[int] = list(range(8, 21))
HIDDEN_DIM: int = 2048  # TinyLlama-1.1B hidden size
# Raw α·v̂ in absolute units stacks across ~13 layers and destabilizes decoding (repetition).
# Scale each layer's injection as a fraction of ||h|| at the last position, with √L scaling.
_STEER_N = max(len(STEER_LAYERS), 1)
_REL_L2_FRAC = 0.05  # α=1 → ~5% of last-position L2 norm per layer (before √L divisor)


# ── Steering vector loading ───────────────────────────────────────────────────

def load_steering_vectors(
    base_path: Path,
    fallback_dim: int = HIDDEN_DIM,
) -> Dict[int, torch.Tensor]:
    """
    Load unit-normalised steering vectors from disk.

    Search order:
      1. <base_path>.pt   — PyTorch dict {layer_idx: Tensor(hidden_dim)}
      2. <base_path>.pkl  — Task 2 MLX output pickle
      3. zero vectors     — no steering effect, but correct structure

    Returns
    -------
    dict mapping layer_idx (int) → unit-norm float32 tensor of shape (hidden_dim,)
    """
    # ── PyTorch format (.pt) ─────────────────────────────────────────────────
    pt_path = base_path.with_suffix(".pt")
    if pt_path.exists():
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        vectors = {int(k): v.float() for k, v in data.items()}
        logger.info("Loaded %d steering vectors from %s", len(vectors), pt_path.name)
        return vectors

    # ── Task 2 pickle format (.pkl) ──────────────────────────────────────────
    pkl_path = base_path.with_suffix(".pkl")
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        vecs_raw = raw.get("vectors", raw) if isinstance(raw, dict) and "vectors" in raw else raw
        vectors = {}
        for layer_idx, val in vecs_raw.items():
            if isinstance(val, dict):
                arr = np.asarray(
                    val.get("unit_vector", val.get("raw_mean_difference")),
                    dtype=np.float32,
                )
            else:
                arr = np.asarray(val, dtype=np.float32)
            arr = arr.ravel()
            norm = np.linalg.norm(arr)
            if norm > 1e-9:
                arr = arr / norm
            vectors[int(layer_idx)] = torch.from_numpy(arr)
        logger.info("Loaded %d steering vectors from %s (pkl)", len(vectors), pkl_path.name)
        return vectors

    # ── Fallback: zero vectors (structure-correct, no steering effect) ────────
    logger.warning(
        "No steering vector file found at %s[.pt|.pkl] — using zero vectors (no effect)",
        base_path,
    )
    return {i: torch.zeros(fallback_dim) for i in STEER_LAYERS}


# ── Steerer ───────────────────────────────────────────────────────────────────

class ActivationSteerer:
    """
    Injects steering vectors into transformer hidden states via forward hooks.

    The injection formula is:
        h' = h + scale * v̂
    where v̂ is the unit-norm steering vector for the given layer.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A loaded causal LM (must expose model.model.layers as a list of
        decoder layers, e.g. LlamaForCausalLM).
    steering_vectors : Dict[int, Tensor]
        {layer_idx: unit-norm tensor of shape (hidden_dim,)}
    """

    def __init__(
        self,
        model,
        steering_vectors: Dict[int, torch.Tensor],
    ) -> None:
        self.model = model
        self.steering_vectors = steering_vectors
        self._hooks: List = []
        self.active: bool = False
        self.scale: float = 1.0
        self.modified_layers: List[int] = []

    # ── hook factory ─────────────────────────────────────────────────────────

    def _make_hook(self, layer_idx: int):
        """Return a forward hook for the given layer index."""

        def hook(module, inputs, output) -> Optional[Tuple]:
            if not self.active:
                return None  # pass-through

            vec = self.steering_vectors.get(layer_idx)
            if vec is None:
                return None

            # transformers ≥4.50 (e.g. 4.57): LlamaDecoderLayer returns a plain
            # Tensor of shape (batch, seq, hidden_dim).
            # transformers <4.50: returns a tuple where output[0] is hidden states.
            # Only the **last** position is steered (next-token logits). Injection uses the
            # **same direction** v̂ but magnitude = (α/√L)·frac·||h|| so it tracks activation scale.
            if isinstance(output, tuple):
                hidden = output[0]   # (batch, seq, hidden_dim)
                steer = vec.to(device=hidden.device, dtype=hidden.dtype)
                last = hidden[:, -1:, :]
                norm = last.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                mag = (float(self.scale) / (_STEER_N ** 0.5)) * _REL_L2_FRAC * norm
                delta = mag * steer.view(1, 1, -1)
                new_h = hidden.clone()
                new_h[:, -1:, :] = last + delta
                self.modified_layers.append(layer_idx)
                return (new_h,) + output[1:]
            else:
                hidden = output
                steer = vec.to(device=hidden.device, dtype=hidden.dtype)
                last = hidden[:, -1:, :]
                norm = last.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                mag = (float(self.scale) / (_STEER_N ** 0.5)) * _REL_L2_FRAC * norm
                delta = mag * steer.view(1, 1, -1)
                new_h = hidden.clone()
                new_h[:, -1:, :] = last + delta
                self.modified_layers.append(layer_idx)
                return new_h

        return hook

    # ── hook management ───────────────────────────────────────────────────────

    def register_hooks(self) -> None:
        """Attach one forward hook per steering layer. Safe to call multiple times."""
        self.remove_hooks()
        decoder_layers = self.model.model.layers
        registered = 0
        for idx in STEER_LAYERS:
            if idx < len(decoder_layers):
                h = decoder_layers[idx].register_forward_hook(self._make_hook(idx))
                self._hooks.append(h)
                registered += 1
        logger.info(
            "Registered %d steering hooks on layers %s",
            registered,
            f"{STEER_LAYERS[0]}–{STEER_LAYERS[-1]}",
        )

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── context manager ───────────────────────────────────────────────────────

    @contextmanager
    def steering_context(self, scale: float = 1.0) -> Generator["ActivationSteerer", None, None]:
        """
        Context manager that enables steering for the duration of a block,
        then restores the disabled state.

        Example
        -------
        with steerer.steering_context(scale=1.5):
            output_ids = model.generate(...)
        layers_hit = steerer.modified_layers   # populated after the block
        """
        self.active = True
        self.scale = scale
        self.modified_layers = []
        try:
            yield self
        finally:
            self.active = False

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of the steerer configuration."""
        n_vecs = len(self.steering_vectors)
        layer_ids = sorted(self.steering_vectors.keys())
        span = f"{layer_ids[0]}–{layer_ids[-1]}" if layer_ids else "none"
        return (
            f"ActivationSteerer | vectors={n_vecs} layers ({span}) | "
            f"hooks_registered={len(self._hooks)} | active={self.active} | scale={self.scale}"
        )

    def __del__(self) -> None:
        self.remove_hooks()
