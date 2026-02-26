#!/usr/bin/env python3
"""Validate Hugging Face Inference API credentials + model callability."""

import os
import sys
from huggingface_hub import InferenceClient


def fail(msg: str, code: int = 1):
    print(f"❌ {msg}")
    sys.exit(code)


def main() -> int:
    token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
    )
    model = os.getenv("HF_INFERENCE_API", "mistralai/Mistral-7B-Instruct-v0.2").strip()
    fallbacks = [
        candidate.strip()
        for candidate in os.getenv("HF_INFERENCE_FALLBACKS", "").split(",")
        if candidate.strip()
    ]
    timeout = float(os.getenv("HF_API_TIMEOUT", "45"))

    if not token:
        fail("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN / HUGGINGFACE_API_TOKEN).")

    if not model:
        fail("HF_INFERENCE_API is empty.")

    prompt = "Reply with exactly: ok"
    candidates = [model, *fallbacks]
    errors = []

    for candidate in candidates:
        print(f"🔎 Checking model callability: {candidate}")
        client = InferenceClient(model=candidate, token=token, timeout=timeout)

        try:
            output = client.text_generation(prompt, max_new_tokens=8, temperature=0.0)
            print(f"✅ Inference succeeded with {candidate}. Sample output: {output!r}")
            print("✅ Hugging Face token + model combination appears callable.")
            return 0
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    fail(
        "Inference call failed for all configured models. This usually means the token is invalid, "
        "the model is not available for your account/provider, or rate limits are hit.\n"
        + "\n".join(errors),
        code=2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
