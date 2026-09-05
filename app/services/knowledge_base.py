"""
Knowledge-base retrieval service.

Retrieval is done with a small, dependency-free TF-IDF + cosine-similarity
implementation (pure Python, no scikit-learn/scipy). This is a deliberate
design choice, for two reasons:

- Reliability (Theme 1 & 3): it is deterministic and fully local, so it
  never fails because of a network/model-availability issue, and it only
  ever returns text that genuinely exists in the knowledge base (grounded
  retrieval) - the agent can never "hallucinate" an answer, it can only
  fail to find one, in which case the caller must escalate or ask for
  clarification instead of guessing.
- Deployability: scikit-learn's dependency, scipy, does not publish
  musllinux wheels, which would force the Alpine-based Docker image to
  compile scipy/BLAS from source - slow, fragile, and a poor fit for a
  small knowledge base that doesn't need a full ML stack. A ~60-line
  TF-IDF implementation covers this use case just as well.

If ``settings.HF_ENABLED`` is true, this module can be swapped/augmented
with semantic (embedding-based) retrieval via Hugging Face - see
``app/services/huggingface_client.py``. TF-IDF remains the guaranteed
fallback so the system degrades gracefully rather than failing outright.
"""

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.support import KnowledgeBaseEntry
from app.services import huggingface_client

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, general-purpose English stopword list - enough to keep common
# function words from dominating short FAQ text, without pulling in a
# corpus/NLP dependency.
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


def _tf(tokens: List[str]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {term: count / total for term, count in counts.items()}


@dataclass
class RetrievedEntry:
    entry: KnowledgeBaseEntry
    relevance: float


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    common_terms = vec_a.keys() & vec_b.keys()
    if not common_terms:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common_terms)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_idf(all_tokens: List[List[str]]) -> Dict[str, float]:
    n_docs = len(all_tokens)
    doc_freq: Counter = Counter()
    for tokens in all_tokens:
        doc_freq.update(set(tokens))
    # Smoothed idf, matching the common scikit-learn-style formulation:
    # idf(t) = ln((1 + N) / (1 + df(t))) + 1
    return {term: math.log((1 + n_docs) / (1 + df)) + 1 for term, df in doc_freq.items()}


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = _tf(tokens)
    return {term: weight * idf.get(term, 0.0) for term, weight in tf.items()}


def _rank(query: str, entries: List[KnowledgeBaseEntry]) -> List[RetrievedEntry]:
    if not entries:
        return []

    doc_tokens = [_tokenize(f"{e.title}. {e.content}") for e in entries]
    query_tokens = _tokenize(query)
    if not query_tokens or not any(doc_tokens):
        return []

    idf = _build_idf(doc_tokens + [query_tokens])
    query_vec = _tfidf_vector(query_tokens, idf)
    if not query_vec:
        return []

    ranked = []
    for entry, tokens in zip(entries, doc_tokens):
        doc_vec = _tfidf_vector(tokens, idf)
        relevance = _cosine_similarity(query_vec, doc_vec)
        ranked.append(RetrievedEntry(entry=entry, relevance=relevance))

    ranked.sort(key=lambda r: r.relevance, reverse=True)
    return ranked


async def _load_entries(
    db: AsyncSession, category: Optional[str] = None
) -> List[KnowledgeBaseEntry]:
    stmt = select(KnowledgeBaseEntry)
    if category:
        stmt = stmt.filter(KnowledgeBaseEntry.category == category)
    result = await db.execute(stmt)
    return list(result.scalars().all())


# --------------------------------------------------------------------------
# BM25 (lexical) - pure Python via rank_bm25, same "no scipy/sklearn" spirit
# as the TF-IDF implementation above: no musllinux wheel problems, fully
# deterministic, and it never fails because of a network/model issue.
# Generally out-performs the TF-IDF cosine baseline on short FAQ-style text.
# --------------------------------------------------------------------------


def _rank_bm25(query: str, entries: List[KnowledgeBaseEntry]) -> List[RetrievedEntry]:
    if not entries:
        return []

    doc_tokens = [_tokenize(f"{e.title}. {e.content}") for e in entries]
    query_tokens = _tokenize(query)
    if not query_tokens or not any(doc_tokens):
        return []

    bm25 = BM25Okapi(doc_tokens)
    raw_scores = bm25.get_scores(query_tokens)

    # BM25 scores are unbounded, unlike the 0..1 cosine scores the rest of
    # the system (confidence.py, KB_MIN_RELEVANCE_SCORE) assumes. Normalize
    # by the max score in this ranking so relevance stays comparable across
    # retrieval modes.
    max_score = max(raw_scores) if len(raw_scores) else 0.0
    ranked = [
        RetrievedEntry(entry=e, relevance=(s / max_score if max_score > 0 else 0.0))
        for e, s in zip(entries, raw_scores)
    ]
    ranked.sort(key=lambda r: r.relevance, reverse=True)
    return ranked


# --------------------------------------------------------------------------
# Dense (semantic) retrieval - optional, only active when HF embeddings are
# enabled. Entry embeddings are computed lazily on first use and cached on
# the row (KnowledgeBaseEntry.embedding) so repeat queries don't re-embed
# the same KB content. Returns [] (never raises) if HF is disabled or a
# call fails, so callers can always fall back to BM25/TF-IDF only.
# --------------------------------------------------------------------------


async def _get_or_compute_embedding(
    db: AsyncSession, entry: KnowledgeBaseEntry
) -> Optional[List[float]]:
    if entry.embedding:
        try:
            return json.loads(entry.embedding)
        except (ValueError, TypeError):
            pass  # corrupt/old cache entry - recompute below

    vector = await huggingface_client.embed_text(f"{entry.title}. {entry.content}")
    if vector is None:
        return None

    entry.embedding = json.dumps(vector)
    db.add(entry)
    await db.flush()
    return vector


def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _rank_dense(
    db: AsyncSession, query: str, entries: List[KnowledgeBaseEntry]
) -> List[RetrievedEntry]:
    if not entries:
        return []

    query_vec = await huggingface_client.embed_text(query)
    if query_vec is None:
        return []  # HF disabled or the call failed - hybrid falls back to BM25 only

    ranked = []
    for entry in entries:
        entry_vec = await _get_or_compute_embedding(db, entry)
        if entry_vec is None:
            continue
        ranked.append(RetrievedEntry(entry=entry, relevance=_cosine(query_vec, entry_vec)))

    ranked.sort(key=lambda r: r.relevance, reverse=True)
    return ranked


# --------------------------------------------------------------------------
# Hybrid retrieval: fuse BM25 (lexical) and dense (semantic) rankings with
# Reciprocal Rank Fusion (RRF) - simple, parameter-light, and doesn't
# require the two scoring scales to be comparable (unlike a weighted-sum
# fusion would). If dense retrieval returns nothing (HF disabled/failed),
# this degrades gracefully to BM25-only, which is itself always available.
# --------------------------------------------------------------------------


def _reciprocal_rank_fusion(
    rankings: List[List[RetrievedEntry]], k: int
) -> List[RetrievedEntry]:
    scores: Dict[int, float] = {}
    entries_by_id: Dict[int, KnowledgeBaseEntry] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking):
            entries_by_id[result.entry.id] = result.entry
            scores[result.entry.id] = scores.get(result.entry.id, 0.0) + 1.0 / (k + rank + 1)

    if not scores:
        return []

    max_score = max(scores.values())
    fused = [
        RetrievedEntry(entry=entries_by_id[eid], relevance=score / max_score)
        for eid, score in scores.items()
    ]
    fused.sort(key=lambda r: r.relevance, reverse=True)
    return fused


async def _rank_hybrid(
    db: AsyncSession, query: str, entries: List[KnowledgeBaseEntry]
) -> List[RetrievedEntry]:
    bm25_results = _rank_bm25(query, entries)
    dense_results = await _rank_dense(db, query, entries)

    if not dense_results:
        # HF embeddings unavailable - BM25-only is still a complete,
        # correct ranking, just without the semantic half.
        return bm25_results

    return _reciprocal_rank_fusion([bm25_results, dense_results], k=settings.KB_RRF_K)


async def search_knowledge_base(
    db: AsyncSession,
    query: str,
    category: Optional[str] = None,
    top_k: Optional[int] = None,
    min_relevance: Optional[float] = None,
    require_verified: bool = False,
    retrieval_mode: Optional[str] = None,
) -> List[RetrievedEntry]:
    """
    Retrieve the most relevant knowledge-base entries for ``query``.

    ``retrieval_mode`` (defaults to ``settings.KB_RETRIEVAL_MODE``) selects
    between "tfidf" (original dependency-free baseline), "bm25" (lexical,
    generally stronger on short FAQ text), and "hybrid" (BM25 + dense
    embeddings via RRF fusion - degrades to BM25-only if HF embeddings are
    unavailable). Every mode always returns *something* if the KB has any
    lexical overlap with the query - retrieval is never a hard dependency
    on an external model.

    Entries below ``min_relevance`` are dropped entirely (source filtering,
    Theme 3) rather than being returned with a low score, so callers never
    accidentally treat noise as a real source.
    """
    top_k = top_k or settings.KB_TOP_K
    min_relevance = (
        min_relevance if min_relevance is not None else settings.KB_MIN_RELEVANCE_SCORE
    )
    mode = retrieval_mode or settings.KB_RETRIEVAL_MODE

    entries = await _load_entries(db, category=category)
    if require_verified:
        entries = [e for e in entries if e.verified]

    if mode == "hybrid":
        ranked = await _rank_hybrid(db, query, entries)
    elif mode == "bm25":
        ranked = _rank_bm25(query, entries)
    else:
        ranked = _rank(query, entries)

    filtered = [r for r in ranked if r.relevance >= min_relevance]
    return filtered[:top_k]
