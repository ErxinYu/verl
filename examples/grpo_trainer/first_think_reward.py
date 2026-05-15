"""
First-Think Efficiency Reward
==============================
Goal: encourage the model to do thorough reasoning in the *first* thinking
      pass before backtracking with transition words (But, Wait, Alternatively…).

Reward formula (per sample):
    score = correctness + first_think_weight * first_think_ratio

where:
    correctness      =  1.0 if answer is correct, -1.0 otherwise
    first_think_ratio = first_think_len / max(1, total_think_len)   ∈ [0, 1]
    first_think_len  = characters from <think> to the first transition word
    total_think_len  = characters inside the full <think>...</think> block

Because GRPO normalizes advantages within a group:
    A_i = (r_i - mean(r)) / std(r)
the relative ranking within a group is recovered automatically — we don't need
explicit rank computation at score time.

Extra metrics logged per sample:
    first_think_len, total_think_len, first_think_ratio,
    first_think_bonus, correctness, acc, pred
"""

import re
from typing import Optional

# Regex patterns for words/phrases that signal the model is **backtracking**
# mid-thought.  Rules of thumb used here:
#
#  1. "Hmm" is excluded — DeepSeek-R1 models routinely open the <think> block
#     with "Hmm, let me …", which is the model *starting* to think, not pivoting.
#     Including it would make first_think_len ≈ 0 for almost every sample.
#
#  2. Short pivot words ("Wait", "But") require a newline prefix so they act as
#     sentence-openers.  Compound phrases like "But wait" / "But hold on" are
#     matched by the "But" line-opener rule and are therefore covered automatically.
#
#  3. Longer explicit phrases (re-examine, reconsider, double-check …) are kept
#     as-is because they are unambiguous regardless of position.
DEFAULT_TRANSITION_PATTERNS: list[str] = [
    r"\n\s*Wait\b",                       # "Wait" at start of a line (incl. "Wait,", "Wait!")
    r"\n\s*But\b",                        # "But" at start of a line (covers "But wait", "But hold on", …)
    r"\bAlternatively\b",
    r"\bActually,\s",                     # "Actually, …" – hedging/reversing a claim
    r"\bHold on\b",
    r"\bIs there another way\b",          # "Is there another way to think about this?"
    r"\bLet me reconsider\b",
    r"\bLet me re-examine\b",
    r"\bLet me try again\b",
    r"\bLet me re-?think\b",
    r"\bLet me double.?check\b",          # "Let me double-check" / "Let me doublecheck"
    r"\bOn second thought\b",
    r"\bI made an error\b",
    r"\bI was wrong\b",
    r"\bAnother approach\b",              # "Another approach would be …"
]


def extract_first_think(
    response_str: str,
    transition_patterns: list[str],
) -> tuple[int, int]:
    """Return (first_think_len, total_think_len) in characters.

    first_think_len  – chars from right after <think> up to (but not including)
                       the first transition word, or end of </think> if none found.
    total_think_len  – chars inside the full <think>…</think> block.
    Both are 0 when no <think> tag is found.
    """
    think_open = response_str.find("<think>")
    if think_open == -1:
        return 0, 0

    content_start = think_open + len("<think>")
    think_close = response_str.find("</think>", content_start)
    think_content = (
        response_str[content_start:think_close]
        if think_close != -1
        else response_str[content_start:]
    )

    total_think_len = len(think_content)

    # Find the earliest transition word
    combined = "|".join(transition_patterns)
    m = re.search(combined, think_content)
    first_think_len = m.start() if m else total_think_len

    return first_think_len, total_think_len


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    # ---- tuneable via reward.custom_reward_function.reward_kwargs ----
    first_think_weight: float = 0.5,
    transition_words: Optional[list[str]] = None,
    **kwargs,
) -> dict:
    """Compute combined correctness + first-think efficiency reward.

    Args:
        data_source:        dataset identifier forwarded to math_dapo scorer.
        solution_str:       decoded model response.
        ground_truth:       reference answer.
        extra_info:         optional extra metadata dict.
        first_think_weight: weight of the first-think bonus (0 → pure correctness).
        transition_words:   optional override for transition-word patterns.

    Returns:
        dict with keys: score, acc, pred,
                        first_think_len, total_think_len, first_think_ratio,
                        first_think_bonus, correctness.
    """
    from verl.utils.reward_score.math_dapo import compute_score as _math_score

    patterns = transition_words if transition_words is not None else DEFAULT_TRANSITION_PATTERNS

    # ── 1. Correctness ────────────────────────────────────────────────────────
    math_result = _math_score(solution_str, ground_truth)
    correctness: float = math_result["score"]   # 1.0 or -1.0
    acc: bool = math_result["acc"]
    pred = math_result.get("pred")

    # ── 2. First-think efficiency ─────────────────────────────────────────────
    first_think_len, total_think_len = extract_first_think(solution_str, patterns)

    first_think_ratio = (
        first_think_len / total_think_len if total_think_len > 0 else 0.0
    )
    first_think_bonus = first_think_weight * first_think_ratio

    # ── 3. Combined score ─────────────────────────────────────────────────────
    total_score = correctness + first_think_bonus

    return {
        "score": total_score,
        "acc": acc,
        "pred": pred,
        "correctness": correctness,
        "first_think_len": first_think_len,
        "total_think_len": total_think_len,
        "first_think_ratio": first_think_ratio,
        "first_think_bonus": first_think_bonus,
    }
