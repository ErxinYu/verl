"""
First-Three-Think Pass@K Reward
================================
Same as first_think_pass4_reward.py, but instead of truncating the CoT at
the **first** backtracking word, we keep the first **three** thinking segments
(i.e. truncate at the 3rd transition-word occurrence).

Reward = correct_count / K   (range [0.0, 1.0])

Intuition
---------
Allows the model three "attempts" inside <think> before we probe whether the
reasoning prefix is sufficient to derive the correct answer.  Comparing this
run against the first-think variant shows how much additional backtracking
actually helps.

Reward kwargs
-------------
  vllm_url    – base URL of the vllm server, e.g. "http://localhost:8001"
  pass_at_k   – number of completions to sample               (default 4)
  temperature – sampling temperature for vllm calls           (default 0.6)
  max_tokens  – max tokens per completion                     (default 512)
  model_name  – model name sent in the vllm request           (default "default")
  n_thinks    – number of thinking segments to keep           (default 3)

Extra metrics logged per sample
--------------------------------
  pass_rate, resp_tokens, first_think_tokens (tokens up to nth boundary),
  original_acc, vllm_error, acc, pred
"""

import asyncio
import re
from functools import lru_cache
from typing import Optional

import aiohttp

DEFAULT_TRANSITION_PATTERNS: list[str] = [
    r"\n\s*Wait\b",
    r"\n\s*But\b",
    r"\bAlternatively\b",
    r"\bActually,\s",
    r"\bHold on\b",
    r"\bIs there another way\b",
    r"\bLet me reconsider\b",
    r"\bLet me re-examine\b",
    r"\bLet me try again\b",
    r"\bLet me re-?think\b",
    r"\bLet me double.?check\b",
    r"\bOn second thought\b",
    r"\bI made an error\b",
    r"\bI was wrong\b",
    r"\bAnother approach\b",
]

_USER_ASSISTANT_PREFIX = "<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n"


@lru_cache(maxsize=1)
def _get_tokenizer(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def _extract_nth_think(response_str: str, patterns: list[str], n: int = 3) -> tuple[str, int]:
    """Return (think_content_up_to_nth_boundary, total_think_len).

    Finds the nth occurrence of any transition pattern inside <think>…</think>
    and truncates there.  If fewer than n transitions exist, returns the full
    thinking content (same as first_think when n=1 and no transition found).
    """
    think_open = response_str.find("<think>")
    if think_open != -1:
        content_start = think_open + len("<think>")
    else:
        content_start = 0

    think_close = response_str.find("</think>", content_start)
    if think_close != -1:
        think_content = response_str[content_start:think_close]
    else:
        think_content = response_str[content_start:]

    if not think_content:
        return "", 0

    combined = "|".join(patterns)
    search_from = 0
    cut_pos = None
    for _ in range(n):
        m = re.search(combined, think_content[search_from:])
        if m is None:
            break
        cut_pos = search_from + m.start()
        search_from = search_from + m.start() + 1  # advance past this match

    truncated = think_content[:cut_pos] if cut_pos is not None else think_content
    return truncated, len(think_content)


def _get_question(extra_info: dict) -> Optional[str]:
    for key in ("question", "problem", "query"):
        val = extra_info.get(key)
        if val and isinstance(val, str):
            return val

    raw_prompt = extra_info.get("raw_prompt")
    if raw_prompt is not None:
        msgs = raw_prompt if isinstance(raw_prompt, list) else list(raw_prompt)
        for msg in msgs:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    return " ".join(texts) or None

    return None


def _build_vllm_prompt(question: str, think_content: str) -> str:
    prefix = _USER_ASSISTANT_PREFIX.format(question=question)
    return prefix + think_content.rstrip() + "\n</think>\n\n"


async def _call_vllm(
    vllm_url: str,
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    model_name: str,
    retries: int = 3,
) -> list[str]:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }
    url = f"{vllm_url.rstrip('/')}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=180)

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            return [choice["text"] for choice in data["choices"]]
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

    return []


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    vllm_url: str = None,
    pass_at_k: int = 4,
    temperature: float = 0.6,
    max_tokens: int = 512,
    model_name: str = "default",
    n_thinks: int = 3,
    transition_words: Optional[list[str]] = None,
    **kwargs,
) -> dict:
    """Compute first-three-think pass@K reward."""
    from verl.utils.reward_score.math_dapo import compute_score as _math_score

    extra_info = extra_info or {}
    patterns = transition_words if transition_words is not None else DEFAULT_TRANSITION_PATTERNS

    math_result = _math_score(solution_str, ground_truth)
    original_correctness: float = math_result["score"]
    original_acc: bool = math_result["acc"]
    pred = math_result.get("pred")

    think_content, _ = _extract_nth_think(solution_str, patterns, n=n_thinks)

    resp_tokens       = _count_tokens(solution_str)
    first_think_tokens = _count_tokens(think_content)

    base_result = {
        "acc": original_acc,
        "pred": pred,
        "original_acc": float(original_acc),
        "resp_tokens": resp_tokens,
        "first_think_tokens": first_think_tokens,
        "vllm_error": 0.0,
    }

    if not vllm_url:
        print(f"[STAT] resp_tok={resp_tokens} think_tok={first_think_tokens} pass_rate={float(original_acc):.4f} original_acc={float(original_acc):.0f} score={original_correctness:.4f} fallback=no_url", flush=True)
        return {**base_result, "score": float(original_acc), "pass_rate": float(original_acc)}

    question = _get_question(extra_info)
    if not question:
        print(f"[STAT] resp_tok={resp_tokens} think_tok={first_think_tokens} pass_rate={float(original_acc):.4f} original_acc={float(original_acc):.0f} score={original_correctness:.4f} fallback=no_question", flush=True)
        return {**base_result, "score": float(original_acc), "pass_rate": float(original_acc)}

    prompt = _build_vllm_prompt(question, think_content)

    try:
        completions = await _call_vllm(
            vllm_url=vllm_url,
            prompt=prompt,
            n=pass_at_k,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=model_name,
        )
    except Exception:
        fallback_score = float(original_acc)
        print(f"[STAT] resp_tok={resp_tokens} think_tok={first_think_tokens} pass_rate={float(original_acc):.4f} original_acc={float(original_acc):.0f} score={fallback_score:.4f} fallback=vllm_error", flush=True)
        return {**base_result, "score": fallback_score, "pass_rate": float(original_acc), "vllm_error": 1.0}

    correct_count = sum(1 for c in completions if _math_score(c, ground_truth)["acc"])
    pass_rate = correct_count / max(len(completions), 1)
    score = float(original_acc) + 0.5 * pass_rate

    result = {**base_result, "score": score, "pass_rate": pass_rate}

    print(
        f"[STAT] resp_tok={resp_tokens}"
        f" think_tok={first_think_tokens}"
        f" pass_rate={pass_rate:.4f}"
        f" original_acc={float(original_acc):.0f}"
        f" score={score:.4f}",
        flush=True,
    )
    return result
