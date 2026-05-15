"""
First-Think Pass@K Reward
==========================
For each rollout response, finds the "first think" boundary (the first
backtracking / self-correction transition word), truncates the chain-of-thought
at that point, closes the </think> tag, and asks a vllm server to generate K
direct-answer completions from that truncated prefix.

Reward = correct_count / K   (range [0.0, 1.0])

Intuition
---------
A model whose *first* reasoning attempt already contains enough information to
derive the correct answer will score higher than one that immediately backtracks
and needs to revise.  GRPO's within-group advantage normalisation turns these
pass-rates into meaningful learning signal.

Requirements
------------
1. A running vllm server (ref model recommended for stability) accessible at
   ``vllm_url``, e.g. "http://localhost:8001".
2. verl's NaiveRewardManager must expose ``raw_prompt`` in extra_info
   (one-line patch to naive.py already applied alongside this file).

Reward kwargs (reward.custom_reward_function.reward_kwargs in the shell script)
--------------------------------------------------------------------------------
  vllm_url    – base URL of the vllm server, e.g. "http://localhost:8001"
  pass_at_k   – number of completions to sample               (default 4)
  temperature – sampling temperature for vllm calls           (default 0.6)
  max_tokens  – max tokens per completion                     (default 512)
  model_name  – model name sent in the vllm request           (default "default")

Extra metrics logged per sample
--------------------------------
  pass_rate, first_think_chars, total_think_chars,
  original_acc, vllm_error, acc, pred
"""

import asyncio
import re
from functools import lru_cache
from typing import Optional

import aiohttp

# Limit concurrent requests to the vllm server to avoid timeout storms
_VLLM_SEMAPHORE = asyncio.Semaphore(64)

# Transition / backtracking words — identical to first_think_reward.py
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

# Prompt prefix for DeepSeek-R1-Distill-Qwen models.
# The chat template ends with <think>\n so the vllm prompt includes the truncated
# think content and </think> after this prefix.
_USER_ASSISTANT_PREFIX = "<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n"


@lru_cache(maxsize=1)
def _get_tokenizer(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_first_think(response_str: str, patterns: list[str]) -> tuple[str, int]:
    """Return (first_think_content, total_think_len).

    Three formats handled:
    1. <think>…</think> in response (rare, skip_special_tokens=False)
    2. No <think> tag, but </think> present: prompt ended with <think>,
       so response IS the thinking content up to </think>, then answer.
    3. No <think> or </think>: response was cut off at max_response_length
       mid-thought; treat entire response as thinking content.
    """
    think_open = response_str.find("<think>")
    if think_open != -1:
        content_start = think_open + len("<think>")
    else:
        content_start = 0  # response starts mid-think

    think_close = response_str.find("</think>", content_start)
    if think_close != -1:
        think_content = response_str[content_start:think_close]
    else:
        # Truncated response — entire response is (partial) thinking
        think_content = response_str[content_start:]

    if not think_content:
        return "", 0

    combined = "|".join(patterns)
    m = re.search(combined, think_content)
    first_think = think_content[: m.start()] if m else think_content
    return first_think, len(think_content)


def _get_question(extra_info: dict) -> Optional[str]:
    """Extract the user question from extra_info, trying several field names."""
    # Direct fields (if dataset includes extra_info with question)
    for key in ("question", "problem", "query"):
        val = extra_info.get(key)
        if val and isinstance(val, str):
            return val

    # raw_prompt injected by NaiveRewardManager from non_tensor_batch
    raw_prompt = extra_info.get("raw_prompt")
    if raw_prompt is not None:
        msgs = raw_prompt if isinstance(raw_prompt, list) else list(raw_prompt)
        for msg in msgs:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # multimodal: list of content blocks
                if isinstance(content, list):
                    texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    return " ".join(texts) or None

    return None


def _build_vllm_prompt(question: str, first_think_content: str) -> str:
    """Build the raw-text prompt fed to /v1/completions.

    The prefix already contains <think>\\n; we append the truncated reasoning
    then close </think> so vllm continues directly to the answer.
    """
    prefix = _USER_ASSISTANT_PREFIX.format(question=question)
    return prefix + first_think_content.rstrip() + "\n</think>\n\n"


async def _call_vllm(
    vllm_url: str,
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    model_name: str,
    retries: int = 3,
) -> list[str]:
    """Call /v1/completions and return n completion strings."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }
    url = f"{vllm_url.rstrip('/')}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=600)

    for attempt in range(retries):
        try:
            async with _VLLM_SEMAPHORE:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
            return [choice["text"] for choice in data["choices"]]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

    return []  # unreachable


# ---------------------------------------------------------------------------
# Main reward function (async)
# ---------------------------------------------------------------------------

async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    # ── tuneable via reward.custom_reward_function.reward_kwargs ──────────
    vllm_url: str = None,
    pass_at_k: int = 4,
    temperature: float = 0.6,
    max_tokens: int = 512,
    model_name: str = "default",
    transition_words: Optional[list[str]] = None,
    **kwargs,
) -> dict:
    """Compute first-think pass@K reward.

    Returns a dict with 'score' (the reward) plus diagnostic fields.
    Falls back to binary correctness reward if vllm is unavailable or the
    question cannot be retrieved.
    """
    from verl.utils.reward_score.math_dapo import compute_score as _math_score

    extra_info = extra_info or {}
    patterns = transition_words if transition_words is not None else DEFAULT_TRANSITION_PATTERNS

    # ── 1. Correctness ────────────────────────────────────────────────────────
    math_result = _math_score(solution_str, ground_truth)
    original_acc: bool = math_result["acc"]
    pred = math_result.get("pred")

    # ── 2. Find first-think boundary ──────────────────────────────────────
    first_think_content, _ = _extract_first_think(solution_str, patterns)

    resp_tokens        = _count_tokens(solution_str)
    first_think_tokens = _count_tokens(first_think_content)

    base_result = {
        "acc": original_acc,
        "pred": pred,
        "original_acc": float(original_acc),
        "resp_tokens": resp_tokens,
        "first_think_tokens": first_think_tokens,
        "vllm_error": 0.0,
    }

    # ── 3. Incorrect responses get reward=0 immediately, skip vllm ───────
    if not original_acc:
        print(f"[STAT] resp_tok={resp_tokens} first_think_tok={first_think_tokens} pass_rate=0.0000 original_acc=0 score=0.0000 skip=incorrect", flush=True)
        return {**base_result, "score": 0.0, "pass_rate": 0.0}

    # ── 4. Guard: need both vllm_url and the question ─────────────────────
    if not vllm_url:
        print(f"[STAT] resp_tok={resp_tokens} first_think_tok={first_think_tokens} pass_rate=0.0000 original_acc=1 score=1.0000 fallback=no_url", flush=True)
        return {**base_result, "score": 1.0, "pass_rate": 0.0}

    question = _get_question(extra_info)
    if not question:
        print(f"[STAT] resp_tok={resp_tokens} first_think_tok={first_think_tokens} pass_rate=0.0000 original_acc=1 score=1.0000 fallback=no_question", flush=True)
        return {**base_result, "score": 1.0, "pass_rate": 0.0}

    # ── 5. Build prompt and call vllm (only for correct responses) ────────
    prompt = _build_vllm_prompt(question, first_think_content)

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
        print(f"[STAT] resp_tok={resp_tokens} first_think_tok={first_think_tokens} pass_rate=0.0000 original_acc=1 score=1.0000 fallback=vllm_error", flush=True)
        return {**base_result, "score": 1.0, "pass_rate": 0.0, "vllm_error": 1.0}

    # ── 6. Score each completion ──────────────────────────────────────────
    correct_count = 0
    for completion in completions:
        if _math_score(completion, ground_truth)["acc"]:
            correct_count += 1

    pass_rate = correct_count / max(len(completions), 1)
    score = 1.0 + 0.5 * pass_rate  # correct=1.0 baseline + first-think bonus [0, 0.5]

    result = {
        **base_result,
        "score": score,
        "pass_rate": pass_rate,
    }

    print(
        f"[STAT] resp_tok={resp_tokens}"
        f" first_think_tok={first_think_tokens}"
        f" pass_rate={pass_rate:.4f}"
        f" original_acc=1"
        f" score={score:.4f}",
        flush=True,
    )
    return result
