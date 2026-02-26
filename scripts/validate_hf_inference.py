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

    text_error = ""

    # 1) Prefer text-generation
    try:
        output = client.text_generation(prompt, max_new_tokens=8, temperature=0.0)
        print(f"✅ Inference succeeded via text-generation. Sample output: {output!r}")
        print("✅ Hugging Face token + model combination appears callable.")
        return 0
    except Exception as text_exc:
        text_error = f"{type(text_exc).__name__}: {text_exc}"
        print(f"⚠️ text-generation not available: {text_error}")

    # 2) Fallback to conversational/chat-completions providers
    try:
        resp = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0.0,
        )
        msg = ""
        if getattr(resp, "choices", None):
            msg = (resp.choices[0].message.content or "").strip()
        print(f"✅ Inference succeeded via conversational/chat_completion. Sample output: {msg!r}")
        print("✅ Hugging Face token + model combination appears callable.")
        return 0
    except Exception as chat_exc:
        fail(
            "Inference call failed for both text-generation and conversational modes. "
            "This usually means token/model/provider availability or rate-limit issues.\n"
            f"text-generation error: {text_error}\n"
            f"conversational error: {type(chat_exc).__name__}: {chat_exc}",
            code=2,
        )
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
