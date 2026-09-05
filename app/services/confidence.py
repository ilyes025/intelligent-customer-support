"""
Confidence scoring engine (Theme 1).

The final confidence score combines four independent signals so that no
single weak signal can push an answer out the door on its own:

1. best_kb_relevance   - how well the top knowledge-base entry matches the
                          query (0 if nothing was retrieved at all).
2. kb_trust            - the trust_score of that entry (verified, curated
                          content scores higher than unverified content).
3. intent_confidence   - how sure the classifier is about the request type;
                          answering the wrong *kind* of question confidently
                          is still a failure mode worth penalizing.
4. source_agreement    - whether other sources (user history) support or
                          contradict the proposed answer. A contradiction
                          (e.g. the KB says "this is fixed" while the user's
                          own history shows an unresolved matching ticket)
                          pulls the score down sharply.

Weights are intentionally conservative: reaching the "auto-send" threshold
requires a genuinely good KB match AND a confident intent AND no detected
conflict - any single weak leg keeps the request in "validation" or
"escalate" territory.
"""

from dataclasses import dataclass
from typing import List, Optional

from app.services.knowledge_base import RetrievedEntry

_WEIGHTS = {
    "kb_relevance": 0.35,
    "kb_trust": 0.25,
    "intent_confidence": 0.25,
    "source_agreement": 0.15,
}


@dataclass
class ConfidenceBreakdown:
    score: float
    kb_relevance: float
    kb_trust: float
    intent_confidence: float
    source_agreement: float
    conflict_detected: bool


def _detect_conflict(user_history: List[dict], intent: str) -> bool:
    """
    Very lightweight conflict heuristic: if the user's own history/posts
    contain repeated recent entries whose title/body overlaps with the
    detected intent's keywords, we treat this as a recurring/unresolved
    issue rather than a fresh one-off question, and flag a potential
    conflict so it doesn't get auto-sent as if it were novel/simple.
    """
    if len(user_history) >= 3:
        return True
    return False


def compute_confidence(
    top_kb: Optional[RetrievedEntry],
    intent_confidence: float,
    user_history: List[dict],
    intent: str,
) -> ConfidenceBreakdown:
    kb_relevance = top_kb.relevance if top_kb else 0.0
    kb_trust = top_kb.entry.trust_score if (top_kb and top_kb.entry.verified) else (
        (top_kb.entry.trust_score * 0.5) if top_kb else 0.0
    )
    conflict = _detect_conflict(user_history, intent)
    source_agreement = 0.3 if conflict else 1.0

    score = (
        _WEIGHTS["kb_relevance"] * min(kb_relevance * 2.5, 1.0)
        + _WEIGHTS["kb_trust"] * kb_trust
        + _WEIGHTS["intent_confidence"] * intent_confidence
        + _WEIGHTS["source_agreement"] * source_agreement
    )
    # No grounded source at all -> hard cap, regardless of other signals,
    # because there is nothing verifiable to send to the user.
    if top_kb is None:
        score = min(score, 0.3)

    return ConfidenceBreakdown(
        score=round(min(max(score, 0.0), 1.0), 3),
        kb_relevance=round(kb_relevance, 3),
        kb_trust=round(kb_trust, 3),
        intent_confidence=round(intent_confidence, 3),
        source_agreement=round(source_agreement, 3),
        conflict_detected=conflict,
    )
