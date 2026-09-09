"""Minimal Ollama REST client for text-only local LLM calls.

pdf_ocr's own OCR/extraction pipeline never needed an LLM — this exists
only for fill_missing_captions.py's caption-recovery pass. Deliberately a
standalone copy rather than importing site_coder's or site_form_segmenter's
identical-shaped client: every repo in this tool family keeps its own copy
rather than sharing a package, and pdf_ocr has no dependency on its
downstream sibling repos today — reaching across to one for a handful of
functions would create exactly that dependency for a small amount of code.
"""
import json
import re

import httpx


def call_ollama(
    system_prompt: str,
    user_content: str,
    model: str,
    base_url: str,
    temperature: float = 0.05,
    timeout: float = 300,
    label: str = "",
) -> str | None:
    """Call Ollama's chat endpoint; return raw text response or None on error."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        # Ollama's per-model default num_ctx (often 2048-4096) is easily
        # exceeded by this prompt's page-window text plus a long verbatim
        # caption — that silently truncates the response mid-JSON rather
        # than erroring, which looks like a model reasoning failure but
        # isn't. Set both explicitly rather than trusting the model default.
        "options": {"temperature": temperature, "num_ctx": 16384, "num_predict": 4096},
    }
    try:
        resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
    except httpx.TimeoutException:
        print(f"  [ERROR] {label}: timed out")
        return None
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        print(f"  [ERROR] {label}: {e}" + (f"\n          {body}" if body else ""))
        return None

    text = ""
    for line in resp.content.splitlines():
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        text += chunk.get("message", {}).get("content", "")
    return text


def extract_json(
    system_prompt: str,
    user_content: str,
    model: str,
    base_url: str,
    temperature: float = 0.05,
    timeout: float = 300,
    label: str = "",
) -> dict | list | None:
    """Call Ollama and parse the response as JSON, tolerating a markdown
    fence or leading/trailing prose (e.g. a reasoning model's think-aloud
    text) around the actual JSON object."""
    text = call_ollama(system_prompt, user_content, model, base_url, temperature, timeout, label)
    if text is None:
        return None

    text = text.strip()
    text = re.sub(r'^```[a-z]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

    preview = text[:600].replace("\n", " ") if text else "<empty>"
    print(f"  [warn] JSON parse failed for {label}  |  response preview: {preview!r}")
    return None
