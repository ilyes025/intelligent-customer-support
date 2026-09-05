"""
Intent & urgency classification (Theme 2: "analyze user context ... classify
and route requests").

Primary classifier is a deterministic keyword scorer: fast, free, always
available, and easy to audit/extend. If Hugging Face is enabled, its
zero-shot result is blended in as a secondary signal; if HF fails or is
disabled, classification silently falls back to the local scorer only, so
intent detection is never a single point of failure.
"""

import re
from dataclasses import dataclass
from typing import Dict, List

from app.services import huggingface_client

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "billing": [
        "invoice", "charge", "charged", "refund", "payment", "price",
        "billing", "subscription", "cancel my order", "money", "overcharged",
    ],
    "shipping": [
        "delivery", "shipping", "shipped", "tracking", "package", "parcel",
        "arrive", "late delivery", "lost package", "courier",
    ],
    "technical": [
        "error", "bug", "not working", "broken", "crash", "login", "password",
        "reset", "install", "app", "website down", "500", "404",
    ],
    "account": [
        "account", "profile", "email address", "username", "delete my account",
        "update my details", "change password", "two-factor",
    ],
    "general": [
        "hours", "hello", "hi", "thank you", "question", "information",
        "policy", "return policy", "contact",
    ],
}

URGENCY_KEYWORDS = {
    "urgent": ["urgent", "asap", "immediately", "emergency", "right now"],
    "high": ["not working", "broken", "can't access", "still waiting", "again"],
}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    urgency: str


def _keyword_scores(text: str) -> Dict[str, float]:
    text_lower = text.lower()
    scores: Dict[str, float] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits:
            # Normalize by keyword-list size so no category is structurally
            # favored just for having more keywords defined.
            scores[intent] = hits / max(len(keywords), 1)
    return scores


def _detect_urgency(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in URGENCY_KEYWORDS["urgent"]):
        return "urgent"
    if any(kw in text_lower for kw in URGENCY_KEYWORDS["high"]):
        return "high"
    return "normal"


def _local_classify(text: str) -> IntentResult:
    scores = _keyword_scores(text)
    urgency = _detect_urgency(text)

    if not scores:
        return IntentResult(intent="general", confidence=0.2, urgency=urgency)

    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]
    # Squash raw hit-ratio into a usable confidence range; a single strong
    # keyword hit should not alone claim near-certainty.
    confidence = min(0.95, 0.4 + best_score * 3)
    return IntentResult(intent=best_intent, confidence=round(confidence, 3), urgency=urgency)


async def classify_intent(text: str) -> IntentResult:
    local_result = _local_classify(text)

    hf_scores = await huggingface_client.zero_shot_classify(
        text, candidate_labels=list(INTENT_KEYWORDS.keys())
    )
    if not hf_scores:
        return local_result

    hf_intent = max(hf_scores, key=lambda k: hf_scores[k])
    hf_confidence = hf_scores[hf_intent]

    if hf_intent == local_result.intent:
        # Both signals agree -> boost confidence, capped at 0.97.
        blended = min(0.97, (local_result.confidence + hf_confidence) / 2 + 0.1)
        return IntentResult(intent=hf_intent, confidence=round(blended, 3), urgency=local_result.urgency)

    # Disagreement between signals: prefer whichever is more confident, but
    # cap the confidence to reflect the uncertainty (Theme 1: don't overstate
    # confidence when sources conflict).
    if hf_confidence > local_result.confidence:
        return IntentResult(
            intent=hf_intent, confidence=round(min(hf_confidence, 0.6), 3), urgency=local_result.urgency
        )
    return IntentResult(
        intent=local_result.intent,
        confidence=round(min(local_result.confidence, 0.6), 3),
        urgency=local_result.urgency,
    )


def is_ambiguous(text: str, intent_confidence: float, threshold: float) -> bool:
    """A message is ambiguous if intent confidence is low and it's short."""
    word_count = len(re.findall(r"\w+", text))
    return intent_confidence < threshold and word_count <= 6
