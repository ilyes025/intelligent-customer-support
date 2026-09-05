"""
Faithfulness / hallucination checking.

The RAG prompt in agent.py *instructs* the model to answer only from the
retrieved context - but an instruction is not a guarantee. This module
verifies that after the fact, by checking whether the generated answer is
actually entailed by the context it was supposed to be grounded in.

Two backends, always with a result:
  1. NLI entailment via the HF zero-shot endpoint (huggingface_client.nli_entailment),
     used when HF is enabled.
  2. A lexical coverage heuristic (fraction of the answer's meaningful
     tokens that also appear in the context) used whenever HF is
     unavailable or fails.

The heuristic is intentionally *not* skipped when HF is unavailable - a
weak faithfulness signal is better than no signal, and it keeps the
routing-downgrade path in agent.py active even in fully offline/no-token
deployments.
"""

import re
from dataclasses import dataclass
from typing import List

from app.core.config import settings
from app.services import huggingface_client

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "for", "with", "about", "as", "by", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "my", "your",
    "his", "her", "our", "their", "can", "will", "would", "should",
    "could", "do", "does", "did", "have", "has", "had", "not", "no",
    "from", "up", "down", "out", "into", "over", "under", "again",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class FaithfulnessResult:
    score: float  # 0..1, higher = better grounded
    is_faithful: bool
    method: str  # "nli" | "lexical_overlap"


def _lexical_coverage(answer: str, context: str) -> float:
    """
    Fraction of the answer's distinct meaningful tokens that also occur in
    the context. Cheap, deterministic, no model call - the guaranteed
    fallback. Not as good as real NLI, but far better than assuming the
    prompt's grounding instruction was followed.
    """
    answer_tokens = set(_tokenize(answer))
    if not answer_tokens:
        return 1.0  # nothing to check against - don't penalize an empty answer here
    context_tokens = set(_tokenize(context))
    covered = answer_tokens & context_tokens
    return len(covered) / len(answer_tokens)


async def check_faithfulness(answer: str, context: str) -> FaithfulnessResult:
    """
    Score how well ``answer`` is supported by ``context``.

    Called from agent.py right after a RAG-generated answer is produced
    (not for raw KB text, which is faithful by construction - it *is* the
    source). A low score downgrades the routing decision one notch
    (auto_send -> pending_validation -> escalated) rather than silently
    shipping a possibly-hallucinated answer.
    """
    if not answer.strip() or not context.strip():
        return FaithfulnessResult(score=0.0, is_faithful=False, method="lexical_overlap")

    nli_score = await huggingface_client.nli_entailment(premise=context, hypothesis=answer)
    if nli_score is not None:
        return FaithfulnessResult(
            score=nli_score,
            is_faithful=nli_score >= settings.FAITHFULNESS_THRESHOLD,
            method="nli",
        )

    score = _lexical_coverage(answer, context)
    return FaithfulnessResult(
        score=score,
        is_faithful=score >= settings.FAITHFULNESS_THRESHOLD,
        method="lexical_overlap",
    )
