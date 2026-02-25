#!/usr/bin/env python3
"""Validate Hugging Face Inference API credentials + model callability."""

import os
import sys
from huggingface_hub import InferenceClient


def fail(msg: str, code: int = 1):
    print(f"❌ {msg}")
    sys.exit(code)


def main() -> int:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    model = os.getenv("HF_INFERENCE_API", "mistralai/Mistral-7B-Instruct-v0.2").strip()
    timeout = float(os.getenv("HF_API_TIMEOUT", "45"))

    if not token:
        fail("Missing HUGGINGFACEHUB_API_TOKEN environment variable.")

    if not model:
        fail("HF_INFERENCE_API is empty.")

    print(f"🔎 Checking model callability: {model}")
    client = InferenceClient(model=model, token=token, timeout=timeout)

    prompt = "Reply with exactly: ok"
    try:
        output = client.text_generation(prompt, max_new_tokens=8, temperature=0.0)
    except Exception as exc:
        fail(
            "Inference call failed. This usually means the token is invalid, the model is not "
            "available for your account/provider, or rate limits are hit.\n"
            f"Error: {type(exc).__name__}: {exc}",
            code=2,
        )

    print(f"✅ Inference succeeded. Sample output: {output!r}")
    print("✅ Hugging Face token + model combination appears callable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
