"""
Extract A/B/C/D from model text for multiple-choice eval.

Avoids a common bug: scanning for the first [ABCD] anywhere matches the first letter of
English words (e.g. "Based" -> B). Option letters are matched only as standalone tokens
(not followed by a lowercase letter, so "Based" does not yield B).
"""

from __future__ import annotations

import re

# "The answer is B", "The best answer is A", "correct option: D"
_EXPLICIT_ANSWER = re.compile(
    r"(?i)(?:the\s+)?(?:best\s+)?(?:correct\s+)?(?:final\s+)?(?:answer|option|choice)\s*(?:is|:)\s*([ABCD])\b"
)
# "it's B", "it is C", "must be A"
_IT_IS_ANSWER = re.compile(
    r"(?i)\b(?:it\s*'?\s*s|must be|should be|therefore\s+is)\s+([ABCD])\b"
)
# "choose B", "select option A"
_CHOOSE = re.compile(r"(?i)\b(?:select|choose|pick)\s+(?:option\s+)?([ABCD])\b")

# Letter must not be part of a word (e.g. "Based" -> B) or an all-caps token ("BASED" -> B).
_STANDALONE_LETTER = re.compile(r"(?<![A-Za-z])([ABCD])(?![a-zA-Z])")


def parse_answer(response: str, choices: list | None = None) -> str:
    """
    Map free-form model output to a single letter in {A,B,C,D}.

    ``choices`` is accepted for API compatibility; letter validation against choices is optional.
    """
    del choices  # reserved for future validation (e.g. only A–C for 3-option items)
    resp = response.strip()
    if not resp:
        return "A"

    # Prefer explicit "answer is X" / "it's X" (use last match — often the conclusion).
    for pattern in (_EXPLICIT_ANSWER, _IT_IS_ANSWER, _CHOOSE):
        matches = list(pattern.finditer(resp))
        if matches:
            return matches[-1].group(1)

    # Strip leading filler so "The best answer is" at line start still works after prefix peel
    lower = resp.lower()
    for prefix in (
        "the best answer is",
        "the correct answer is",
        "the answer is",
        "the best option is",
        "the correct option is",
        "best answer:",
        "best option:",
    ):
        if lower.startswith(prefix):
            resp = resp[len(prefix) :].lstrip()
            lower = resp.lower()
            break

    # First line alone if it is only a letter (optional . ) — strong signal
    first_line = resp.split("\n", 1)[0].strip()
    if len(first_line) <= 4:
        m0 = re.match(r"^([ABCD])[\.\)]?\s*$", first_line.upper())
        if m0:
            return m0.group(1)

    # Leading option letter only if not the start of an English word (e.g. not "Based…").
    c0 = resp[0].upper()
    if c0 in "ABCD" and len(resp) == 1:
        return c0
    if c0 in "ABCD" and len(resp) > 1:
        c1 = resp[1]
        if (not c1.isalpha()) or c1.isspace():
            return c0

    tokens = _STANDALONE_LETTER.findall(resp)
    if len(tokens) == 1:
        return tokens[0]
    if len(tokens) > 1:
        # Prefer the last standalone letter (model often hedges then concludes).
        return tokens[-1]

    # Do not fall back to scanning any [ABCD] in the string — that matches letters inside words (e.g. "Based").
    return "A"
