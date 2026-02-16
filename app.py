import streamlit as st
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler  # type: ignore

# --- Configuration ---
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = Path("./adapters")
LOG_FILE = Path("./logs/chat_log.jsonl")
LOG_FILE.parent.mkdir(exist_ok=True, parents=True)

st.set_page_config(
    page_title="TinyLlama MLX Chat",
    page_icon="tao",
    layout="centered"
)

# --- Custom CSS for Premium Feel ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stChatMessage {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stTextInput input {
        background-color: #262730;
        color: #ffffff;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.current_selection = None

# --- Helper Functions ---
def get_available_versions() -> Dict[str, Dict[str, Optional[str]]]:
    """Scan for available models and adapters."""
    versions: Dict[str, Dict[str, Optional[str]]] = {"Base Model": {"type": "base", "path": None}}
    
    # Scan Adapters (LoRA/QLoRA)
    if ADAPTER_DIR.exists():
        for d in ADAPTER_DIR.iterdir():
            if d.is_dir():
                versions[f"Adapter: {d.name}"] = {"type": "adapter", "path": str(d)}  # type: ignore

    # Scan Full Models
    models_dir = Path("./models")
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir() and "full" in d.name: # Simple filter for full models
                 versions[f"Full Model: {d.name}"] = {"type": "full", "path": str(d)}  # type: ignore
    
    return versions

def load_model_cached(selection_name, version_info):
    """Load model based on selection type."""
    if st.session_state.model is None or st.session_state.current_selection != selection_name:
        
        # Clear previous model to free memory (important for full models)
        if st.session_state.model is not None:
             del st.session_state.model
             del st.session_state.tokenizer
             if hasattr(mx, "metal"):
                 mx.metal.clear_cache()
        
        with st.spinner(f"Loading {selection_name}..."):
            
            model_type = version_info["type"]
            path = version_info["path"]
            
            if model_type == "base":
                st.session_state.model, st.session_state.tokenizer = load(MODEL_PATH)
            
            elif model_type == "adapter":
                # LoRA/QLoRA: Load base model + adapter
                st.session_state.model, st.session_state.tokenizer = load(
                    MODEL_PATH,
                    adapter_path=path
                )
            
            elif model_type == "full":
                # Full FT in mlx-lm still produces adapter files, not a fused model.
                # Load base model + adapter path, same as LoRA/QLoRA.
                st.session_state.model, st.session_state.tokenizer = load(
                    MODEL_PATH,
                    adapter_path=path
                )
            
            st.session_state.current_selection = selection_name
            st.toast(f"Loaded: {selection_name}", icon="✅")

def log_interaction(prompt, response, stats):
    """Log interaction to JSONL file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model_version": st.session_state.current_selection,
        "prompt": prompt,
        "response": response,
        "stats": stats
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# --- UI Layout ---
st.title("🦙 TinyLlama-1.1B MLX Playground")
st.caption("Experiments: LoRA Ranks, Safety Prompts & Model Comparison")

# Sidebar
with st.sidebar:
    st.header("实验 Controls")
    
    # Mode Selection
    compare_mode = st.toggle("⚔️ Compare Mode", value=False, help="Compare two models side-by-side")
    safety_mode = st.toggle("🛡️ Safety Mode", value=False, help="Inject safety system prompt")
    
    st.divider()
    
    # Version Selector
    versions = get_available_versions()
    version_keys = list(versions.keys())
    
    if compare_mode:
        st.subheader("Model A")
        selection_a = st.selectbox("Select Left Model", options=version_keys, index=0, key="model_a")
        
        st.subheader("Model B")
        selection_b = st.selectbox("Select Right Model", options=version_keys, index=min(1, len(version_keys)-1), key="model_b")
        
        if st.button("Load Both Models"):
            # Load Model A
            load_model_cached(selection_a, versions[selection_a])
            # We need to handle two models in memory. 
            # Current load_model_cached assumes single model in session_state.
            # We will patch this below in valid python logic, but for simplicity in this specific app structure:
            # We will toggle 'current_model' reference when generating.
            # This is tricky with single GPU memory. MLX shares memory well, but let's see.
            # actually, for a smooth comparison, we might need to load/unload or hold both if valid.
            # For 1.1B models, holding 2 in memory (approx 4-5GB total) is fine on most Macs (8GB+).
            st.toast("Comparison Ready (Loading happens on generation if needed)", icon="⚔️")
    else:
        selection_name = st.selectbox(
            "Select Model Version",
            options=version_keys,
            index=0
        )
        if st.button("Load Model"):
            load_model_cached(selection_name, versions[selection_name])

    st.divider()
    st.subheader("Generation Params")
    temp = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 64, 512, 256)

# --- Logic for Comparison ---
# To keep it simple and stable, we will modify load_model_cached to store models in a dictionary if in compare mode,
# or just manage the single global one. 
# actually, let's just make 'generate_response' function handles loading if the requested model isn't the active one.

def ensure_model_loaded(model_name, version_data):
    """Ensure the specific model is loaded in session_state.current_model_name"""
    if st.session_state.get("current_selection") != model_name:
        load_model_cached(model_name, version_data)

# Safety System Prompt
SAFETY_PROMPT = """You are a helpful and harmless assistant. You must answer truthfully and politely. 
Important Safety Rules:
1. Do not generate hate speech, violence, or illegal content.
2. If asked to do something harmful, politely refuse and explain why.
3. Keep responses respectful and constructive.
"""

def format_prompt(user_prompt, tokenizer):
    messages = []
    if safety_mode:
        messages.append({"role": "system", "content": SAFETY_PROMPT})
    
    # Add history? For comparison simplicity, let's stick to single turn or simplified history
    # Adding simplified history from session state if needed, but let's do single turn for cleaner comparison
    messages.append({"role": "user", "content": user_prompt})
    
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback
        sys = f"System: {SAFETY_PROMPT}\n" if safety_mode else ""
        return f"{sys}User: {user_prompt}\nAssistant:"

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant": # Legacy single mode
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    elif msg["role"] == "comparison": # New comparison mode
        cols = st.columns(2)
        with cols[0]:
            st.info(f"**{msg['model_a']}**")
            st.markdown(msg['content_a'])
        with cols[1]:
            st.info(f"**{msg['model_b']}**")
            st.markdown(msg['content_b'])

if prompt := st.chat_input("Ask TinyLlama (Experiment Mode)..."):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generation
    if compare_mode:
        # Side-by-Side Generation
        cols = st.columns(2)
        
        # Model A
        with cols[0]:
            st.caption(f"Waiting for {selection_a}...")
            ensure_model_loaded(selection_a, versions[selection_a])
            
            prompt_fmt = format_prompt(prompt, st.session_state.tokenizer)
            sampler = make_sampler(temp)
            
            start = time.time()
            response_a = generate(
                st.session_state.model, 
                st.session_state.tokenizer, 
                prompt=prompt_fmt, 
                max_tokens=max_tokens, 
                sampler=sampler,
                verbose=False
            )
            dur_a = time.time() - start
            st.markdown(response_a)
            st.caption(f"⏱️ {dur_a:.2f}s")

        # Model B
        with cols[1]:
            st.caption(f"Waiting for {selection_b}...")
            ensure_model_loaded(selection_b, versions[selection_b])
            
            prompt_fmt = format_prompt(prompt, st.session_state.tokenizer)
            sampler = make_sampler(temp)
            
            start = time.time()
            # Note: We regenerate prompt_fmt because tokenizer might (rarely) differ if switching base models
            response_b = generate(
                st.session_state.model, 
                st.session_state.tokenizer, 
                prompt=prompt_fmt, 
                max_tokens=max_tokens, 
                sampler=sampler,
                verbose=False
            )
            dur_b = time.time() - start
            st.markdown(response_b)
            st.caption(f"⏱️ {dur_b:.2f}s")
            
        # Log to history
        st.session_state.messages.append({
            "role": "comparison", 
            "model_a": selection_a,
            "content_a": response_a,
            "model_b": selection_b,
            "content_b": response_b
        })
        
    else:
        # Single Mode (Classic)
        ensure_model_loaded(selection_name, versions[selection_name])
        
        with st.chat_message("assistant"):
            prompt_fmt = format_prompt(prompt, st.session_state.tokenizer)
            sampler = make_sampler(temp)
            
            start = time.time()
            response = generate(
                st.session_state.model, 
                st.session_state.tokenizer, 
                prompt=prompt_fmt, 
                max_tokens=max_tokens, 
                sampler=sampler, 
                verbose=False
            )
            dur = time.time() - start
            
            st.markdown(response)
            st.caption(f"⏱️ {dur:.2f}s")
            
            st.session_state.messages.append({"role": "assistant", "content": response})
