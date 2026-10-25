"""Shared helpers: LLM client (any OpenAI-compatible API), JSON parsing, mock mode."""
import json
import os
import random
import re
import time

from dotenv import load_dotenv

load_dotenv()

MOCK = os.getenv("MOCK", "0") == "1"


class LLM:
    """Thin wrapper around any OpenAI-compatible endpoint.

    Set in .env:
      API_KEY=...
      BASE_URL=https://openrouter.ai/api/v1   (or https://api.openai.com/v1, Gemini/Mistral/DeepSeek compat URLs)
    """

    def __init__(self, model, temperature=0.7, base_url=None, reasoning_effort="none"):
        self.model = model
        self.temperature = temperature
        # "none" turns off thinking on Modal-hosted models; pass "omit" to not send the field
        self.extra = {} if reasoning_effort in (None, "omit") else {"reasoning_effort": reasoning_effort}
        if not MOCK:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["API_KEY"],
                                 base_url=base_url or os.getenv("BASE_URL") or None)

    def ask_json(self, system, user, retries=4):
        if MOCK:
            return mock_response(system, user)
        last_err = None
        for attempt in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=4096,
                    extra_body=self.extra,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                msg = r.choices[0].message
                text = msg.content or ""
                if not text.strip():
                    # some models put the answer (or all output) in a separate reasoning field
                    extra = getattr(msg, "model_extra", None) or {}
                    text = (extra.get("reasoning_content") or extra.get("reasoning")
                            or getattr(msg, "reasoning_content", None) or "")
                if not text.strip():
                    raise ValueError(f"Empty response (finish_reason={r.choices[0].finish_reason})")
                return parse_json(text)
            except Exception as e:  # network / rate limit / bad JSON
                last_err = e
                print(f"    retry {attempt + 1}/{retries}: {str(e)[:150]}")
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {retries} tries: {last_err}")


def parse_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"No JSON in response: {text[:200]}")
    return json.loads(m.group(0))


# ---------------------------------------------------------------- mock mode
def mock_response(system, user):
    """Plausible fake outputs so the whole pipeline can be tested for free."""
    if "MONITOR" in system:
        return {"collusion_probability": random.randint(0, 100),
                "suspected_pair": random.sample(["Alex", "Blake", "Casey", "Drew"], 2),
                "reasoning": "mock"}
    secret = "SECRET PACT" in system or "PRIVATE GOAL" in system
    choice = "EXPENSIVE" if (secret and random.random() < 0.8) or random.random() < 0.3 else "CHEAP"
    if "Stage 3" in user:
        return {"choice": choice, "reasoning": "mock choice"}
    if "Stage 2" in user:
        return {"announced_choice": "CHEAP", "message": "Let's all go cheap and keep the bill low."}
    return {"intended_choice": choice, "reasoning": "mock plan"}