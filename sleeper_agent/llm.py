"""One chat call for every backend: anthropic, openai-compat, claude-cli, or none.

    text, err = chat(system, [{"role": "user", "content": "..."}], cfg, phase="draft")

`cfg` is the `llm` config block. A phase names a set of keys: `<phase>_model`,
`<phase>_timeout_s`, `<phase>_effort`. `err` is empty on success. `session` keeps one claude-cli
conversation across calls; the other backends take the whole message list each time.
"""
from __future__ import annotations

import os
import shutil
import subprocess

DEFAULT_TIMEOUT_S = {"draft": 90.0, "manage": 180.0}
MAX_TOKENS = {"draft": 1200, "manage": 4000}


def resolve(cfg: dict, phase: str) -> tuple[str, str, float, str]:
    cfg = cfg or {}
    timeout = cfg.get(f"{phase}_timeout_s") or DEFAULT_TIMEOUT_S.get(phase, 90.0)
    return (cfg.get("backend", "none"), cfg.get(f"{phase}_model") or "", float(timeout),
            cfg.get(f"{phase}_effort") or "")


def _openai_compat(system: str, messages: list[dict], cfg: dict, model: str, timeout: float) -> str:
    import requests
    base = (cfg.get("base_url") or "").rstrip("/")
    if not base:
        raise RuntimeError("llm.base_url not set (e.g. https://openrouter.ai/api/v1)")
    key = os.environ.get(cfg.get("api_key_env", "LLM_API_KEY"), "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = requests.post(f"{base}/chat/completions", headers=headers, timeout=timeout,
                      json={"model": model,
                            "messages": [{"role": "system", "content": system}] + list(messages)})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def chat(system: str, messages: list[dict], cfg: dict, phase: str = "draft",
         session: str | None = None) -> tuple[str, str]:
    backend, model, timeout, effort = resolve(cfg, phase)
    try:
        if backend == "anthropic":
            import anthropic  # noqa
            client = anthropic.Anthropic()
            kw = {"output_config": {"effort": effort}} if effort else {}
            msg = client.messages.create(model=model, max_tokens=MAX_TOKENS.get(phase, 1200), system=system,
                                         messages=list(messages), **kw)
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip(), ""
        if backend == "openai-compat":
            return _openai_compat(system, messages, cfg, model, timeout).strip(), ""
        if backend == "claude-cli":
            exe = shutil.which("claude")
            if not exe:
                return "", "claude CLI not found on PATH"
            args = [exe, "-p", "--output-format", "text", "--tools", ""]
            if effort:  # -- unset inherits the user-level session default, which may be slow
                args += ["--effort", effort]
            if model:
                args += ["--model", model]
            args += ["--system-prompt", system]
            # -- the CLI keeps history under the session id, so only the last turn goes on stdin
            if session:
                args += ["--session-id" if len(messages) == 1 else "--resume", session]
            res = subprocess.run(args, input=messages[-1]["content"], capture_output=True, text=True,
                                 timeout=timeout)
            if res.returncode != 0:
                return "", f"claude CLI exit {res.returncode}: {res.stderr[:120]}"
            return res.stdout.strip(), ""
        return "", "llm disabled"
    except Exception as e:  # noqa
        return "", f"{type(e).__name__}: {str(e)[:120]}"
