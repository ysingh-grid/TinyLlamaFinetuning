"""Static HTML + PNG heatmaps for attention rollout and token attribution."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import base64
import html
import os
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _truncate(s: str, n: int = 18) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 2] + "…"


def tokens_for_plot(tokenizer, input_ids: Sequence[int]) -> List[str]:
    ids = list(input_ids)
    return [tokenizer.convert_ids_to_tokens(i) for i in ids]


def plot_rollout_heatmap(
    rollout: np.ndarray,
    tokens: List[str],
    prompt_len: int,
    title: str,
    out_path: Path,
    max_side: int = 48,
) -> None:
    """Save rollout [S,S] as PNG; vertical line at prompt boundary."""
    s = rollout.shape[0]
    if s > max_side:
        # Downsample for display
        idx = np.linspace(0, s - 1, max_side).astype(int)
        r = rollout[np.ix_(idx, idx)]
        toks = [tokens[i] for i in idx]
        pl = int(prompt_len * max_side / s)
    else:
        r = rollout
        toks = tokens
        pl = prompt_len

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(r, cmap="magma", aspect="auto", vmin=0, vmax=r.max() + 1e-8)
    ax.axvline(pl - 0.5, color="cyan", linewidth=1, alpha=0.8)
    ax.axhline(pl - 0.5, color="cyan", linewidth=1, alpha=0.8)
    tick = range(0, len(toks), max(1, len(toks) // 16))
    ax.set_xticks(list(tick))
    ax.set_xticklabels([_truncate(str(toks[i]), 12) for i in tick], rotation=65, ha="right", fontsize=6)
    ax.set_yticks(list(tick))
    ax.set_yticklabels([_truncate(str(toks[i]), 12) for i in tick], fontsize=6)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_attribution_bar(
    attribution: np.ndarray,
    tokens: List[str],
    prompt_len: int,
    title: str,
    out_path: Path,
) -> None:
    """
    attribution: length S — influence of each input position on last output (or rollout row).
    """
    s = min(len(attribution), len(tokens))
    a = attribution[:s]
    toks = tokens[:s]
    fig, ax = plt.subplots(figsize=(12, 3))
    colors = ["#2ecc71" if i < prompt_len else "#3498db" for i in range(s)]
    ax.bar(range(s), a, color=colors, width=0.9)
    ax.set_xticks(range(0, s, max(1, s // 24)))
    ax.set_xticklabels([_truncate(str(toks[i]), 10) for i in range(0, s, max(1, s // 24))], rotation=75, ha="right", fontsize=5)
    ax.set_title(title)
    ax.set_ylabel("rollout mass")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def png_to_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _img_src_for_html(out_path: Path, image_ref: str, embed_images: bool) -> str:
    """Return src for <img>: data URI if embed_images, else path relative to HTML file."""
    if image_ref.startswith("data:"):
        return image_ref
    p = Path(image_ref).resolve()
    html_dir = out_path.parent.resolve()
    if embed_images:
        return png_to_data_uri(p)
    try:
        rel = p.relative_to(html_dir)
    except ValueError:
        rel = Path(os.path.relpath(p, html_dir))
    return rel.as_posix()


def build_combined_html(
    sections: List[tuple[str, str, str]],
    out_path: Path,
    extra_md: str = "",
    embed_images: bool = False,
) -> None:
    """
    sections: list of (heading, description, image_path_or_data_uri)

    By default **embed_images=False**: write small HTML that references PNGs by **relative path**
    (e.g. ``heatmaps/rollout_0.png``). This is the usual static-report pattern: readable, diffable,
    and works in browsers without huge data-URI limits. Use **embed_images=True** only for a
    single portable file (can exceed ~400KB and break Gradio ``gr.HTML`` previews).
    """
    lines: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "  <title>Attention visualization — Task 3</title>",
        "  <style>",
        "    body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }",
        "    h1 { font-size: 1.5rem; }",
        "    h2 { border-bottom: 1px solid #ccc; margin-top: 2rem; font-size: 1.15rem; }",
        "    figure { margin: 1rem 0; }",
        "    img { max-width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 4px; }",
        "    .meta { background: #f6f8fa; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1.5rem; }",
        "    .caption { color: #444; font-size: 0.95rem; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>TinyLlama LoRA — Attention rollout &amp; attribution</h1>",
        f"  <div class=\"meta\">{extra_md}</div>",
    ]

    for head, desc, img in sections:
        src = _img_src_for_html(out_path, img, embed_images)
        lines.append(f"  <h2>{html.escape(head)}</h2>")
        lines.append(f"  <p class=\"caption\">{html.escape(desc)}</p>")
        lines.append("  <figure>")
        lines.append(
            f"    <img src=\"{html.escape(src, quote=True)}\" "
            f"alt=\"{html.escape(head, quote=True)}\" loading=\"lazy\"/>"
        )
        lines.append("  </figure>")

    lines.extend(["</body>", "</html>", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")
