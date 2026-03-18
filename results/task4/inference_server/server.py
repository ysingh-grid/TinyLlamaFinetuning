
from fastapi import FastAPI
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load, generate

app = FastAPI(title="TinyLlama Edge Inference")
model, tokenizer = load("/Users/ysingh/PyCharmMiscProject/results/task4/model_q4")

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
def generate_text(req: GenerationRequest):
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template([
            {"role": "user", "content": req.prompt}
        ], tokenize=False, add_generation_prompt=True)
    else:
        prompt = req.prompt + "\nAssistant: "
        
    res = generate(model, tokenizer, prompt, max_tokens=req.max_tokens, verbose=False)
    return {"response": res}
