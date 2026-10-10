"""A small, VERIFIABLE instruction-following task for supervised fine-tuning (SFT).

Prompt:
    Write a short story.
    Words: ball, happy, garden
    Dialogue: yes
    Story:
    <model writes the story here>

Constraints are derived from a real story, so a valid answer always exists, and they can be
checked programmatically -> we can report a constraint-following rate before/after SFT, and the
same check can later serve as a reward function for RL.
"""
from __future__ import annotations

import re

STOPWORDS = set(
    "that this with they them their there then than were what when where which will would about after "
    "again also because been before being could does doing down each even every from have here into just "
    "like made make many more most much must only other over said same should some such take than very "
    "want went your".split()
)
QUOTES = ('"', "“", "”")


def has_dialogue(text: str) -> bool:
    return any(q in text for q in QUOTES)


def extract_constraints(story: str, rng, n_words: int = 3):
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", story)}
    cands = sorted(w for w in words if 4 <= len(w) <= 9 and w not in STOPWORDS)
    if len(cands) < n_words:
        return None
    return {"words": rng.sample(cands, n_words), "dialogue": has_dialogue(story)}


def build_prompt(c: dict) -> str:
    return (
        "Write a short story.\n"
        f"Words: {', '.join(c['words'])}\n"
        f"Dialogue: {'yes' if c['dialogue'] else 'no'}\n"
        "Story:\n"
    )


def check(story: str, c: dict) -> dict:
    low = story.lower()
    hits = [bool(re.search(rf"\b{re.escape(w)}\b", low)) for w in c["words"]]
    dialogue_ok = has_dialogue(story) == c["dialogue"]
    return {
        "word_hits": sum(hits),
        "n_words": len(hits),
        "all_words": all(hits),
        "dialogue_ok": dialogue_ok,
        "success": all(hits) and dialogue_ok,
    }