"""Serve an open-weights model on Modal as an OpenAI-compatible endpoint (vLLM).

Adapted from Modal's vLLM inference example (modal.com/docs/examples/llm_inference).

  pip install modal && modal setup          # one-time login
  modal deploy modal_server.py              # prints the server URL
  modal app stop collusion-llm              # STOP IT when done, so it stops billing

Then in .env:
  BASE_URL=<printed URL>/v1
  API_KEY=not-needed
and pass --model llm to the run scripts.
"""
import subprocess

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PORT = 8000

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("collusion-llm")


@app.server(
    image=vllm_image,
    gpu="L40S:1",                 # 48 GB: plenty for a 7B model, much cheaper than H100/H200
    scaledown_window=10 * 60,     # shuts down after 10 idle minutes
    startup_timeout=15 * 60,      # first boot downloads ~15 GB of weights
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    port=PORT,
    target_concurrency=64,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        self.process = subprocess.Popen([
            "vllm", "serve", MODEL,
            "--served-model-name", MODEL, "llm",
            "--host", "0.0.0.0", "--port", str(PORT),
            "--max-model-len", "8192",
            "--enforce-eager",        # faster cold start
        ])

    @modal.exit()
    def stop(self):
        self.process.terminate()
