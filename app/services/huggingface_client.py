"""
Optional Hugging Face client: hosted inference for classification/NLI/
generation, plus a locally-run embedding model.

The hosted-API functions (``zero_shot_classify``, ``nli_entailment``,
``generate_reply``) are deliberately isolated behind ``settings.HF_ENABLED``
(true only when ``HF_API_TOKEN`` is configured). ``embed_text`` runs
entirely locally via `sentence-transformers` and has no such dependency -
see its docstring for why. The rest of the system must work correctly
without any of this - see ``app/services/classifier.py`` for the local
fallback used when Hugging Face is disabled or a call fails. This keeps the
agent reliable even if the external model API is unavailable, rate-limited,
or simply not configured (e.g. in CI/tests).
"""

from typing import Dict, List, Optional

import asyncio

import httpx

from app.core.config import settings

# HF's old flat "router.huggingface.co/models/{model}" URL was retired along
# with the move to "Inference Providers". Task-based calls (zero-shot
# classification, NLI) now live under the hf-inference provider path.
# See diagnose_hf.py for how this was diagnosed.
_API_ROOT = "https://router.huggingface.co/hf-inference/models"
_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"


def _parse_zero_shot_response(data: object) -> Optional[Dict[str, float]]:
    """
    Normalize a zero-shot-classification response into ``{label: score}``.

    hf-inference currently returns a LIST of per-label objects, e.g.
    ``[{"label": "weather", "score": 0.89}, {"label": "sports", "score": 0.10}]``
    - sorted by score, not necessarily in the original candidate_labels
    order. Some older deployments/providers instead return a single dict
    shaped ``{"labels": [...], "scores": [...]}``. Handle both rather than
    assuming one, since this API's response shape isn't stable across
    providers or over time (see diagnose_hf.py's notes on the router).
    """
    if isinstance(data, list):
        result: Dict[str, float] = {}
        for item in data:
            if not isinstance(item, dict):
                return None
            label = item.get("label")
            score = item.get("score")
            if label is None or score is None:
                return None
            result[label] = score
        return result or None

    if isinstance(data, dict):
        labels = data.get("labels")
        scores = data.get("scores")
        if labels and scores and len(labels) == len(scores):
            return dict(zip(labels, scores))
        return None

    return None


async def zero_shot_classify(
    text: str, candidate_labels: List[str]
) -> Optional[Dict[str, float]]:
    """
    Call a Hugging Face zero-shot-classification model.

    Returns a mapping ``{label: score}`` or ``None`` on any failure/timeout,
    so callers can fall back to the local classifier without special-casing
    exceptions.
    """
    if not settings.HF_ENABLED:
        return None

    url = f"{_API_ROOT}/{settings.HF_ZERO_SHOT_MODEL}"
    headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
    payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels}}

    try:
        async with httpx.AsyncClient(timeout=settings.HF_API_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    return _parse_zero_shot_response(data)


_local_embedder = None
_local_embedder_load_failed = False


def _get_local_embedder():
    """
    Lazily load and cache the local sentence-transformers model.

    Returns ``None`` (and remembers not to retry) if the package isn't
    installed or the model fails to load - same "optional enhancement"
    contract as the rest of this module. Loaded at most once per process.
    """
    global _local_embedder, _local_embedder_load_failed
    if _local_embedder is not None:
        return _local_embedder
    if _local_embedder_load_failed:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _local_embedder = SentenceTransformer(settings.HF_EMBEDDING_MODEL)
        return _local_embedder
    except Exception:
        # Covers ImportError (package not installed) as well as any
        # download/load failure - embeddings are an optional enhancement,
        # never a hard dependency.
        _local_embedder_load_failed = True
        return None


def _encode_local(text: str) -> Optional[List[float]]:
    model = _get_local_embedder()
    if model is None:
        return None
    try:
        vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return [float(x) for x in vector]
    except Exception:
        return None


async def embed_text(text: str) -> Optional[List[float]]:
    """
    Compute a sentence embedding locally via `sentence-transformers`.

    Returns a dense vector, or ``None`` on any failure (package missing,
    model load failure, encoding error) - exactly the same contract as
    ``zero_shot_classify`` and ``generate_reply``. Callers (knowledge_base.py's
    hybrid search) must always have a non-embedding fallback ready: hybrid
    retrieval degrades to BM25-only when this returns None.

    This runs locally rather than calling HF's hosted Inference API: HF has
    been progressively dropping free-tier serverless hosting for
    sentence-transformer models, which made the API path unreliable. Running
    locally also removes HF_API_TOKEN as a dependency for this feature
    entirely and drops a network round-trip from the retrieval hot path.
    The blocking encode() call is offloaded to a thread so it doesn't stall
    the event loop.
    """
    return await asyncio.to_thread(_encode_local, text)


async def nli_entailment(premise: str, hypothesis: str) -> Optional[float]:
    """
    Score whether ``hypothesis`` is entailed by ``premise`` using a
    zero-shot NLI model, returning P(entailment) in 0..1, or ``None`` on
    any failure/timeout/disabled state.

    Used by app/services/faithfulness.py to check whether a generated
    answer is actually supported by its retrieved knowledge-base context,
    rather than trusting the RAG prompt's instructions alone.
    """
    if not settings.HF_ENABLED:
        return None

    url = f"{_API_ROOT}/{settings.HF_NLI_MODEL}"  # now correctly under /hf-inference/models/
    headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
    # bart-large-mnli style zero-shot endpoint: treat the premise as the
    # text to classify and "This is supported by the source material." /
    # "This contradicts or invents beyond the source material." as the
    # candidate labels - a standard trick for repurposing a zero-shot
    # classification endpoint as a lightweight entailment check without
    # needing a separate raw-NLI inference path.
    payload = {
        "inputs": f"{hypothesis}\n\nContext: {premise}",
        "parameters": {
            "candidate_labels": [
                "supported by the context",
                "not supported by the context",
            ]
        },
    }

    try:
        async with httpx.AsyncClient(timeout=settings.HF_API_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    label_scores = _parse_zero_shot_response(data)
    if not label_scores:
        return None
    return label_scores.get("supported by the context")


async def generate_reply(prompt: str) -> Optional[str]:
    """
    Call a Hugging Face chat-completion model via the router's
    OpenAI-compatible endpoint.

    Returns the generated text, or ``None`` on any failure/timeout/disabled
    state, exactly like ``zero_shot_classify`` - callers must always have a
    non-LLM fallback ready (see ``app/services/agent.py``: RAG answers fall
    back to raw KB text, and ungrounded drafts fall back to ``None``, which
    the agent treats as "no draft available" rather than surfacing an error).

    Uses /v1/chat/completions rather than the old raw text-generation task
    URL: chat-style models are now expected to be called this way, and the
    old endpoint no longer resolves for most current models. If
    HF_GENERATION_PROVIDER is set, it's appended as a ":<provider>" suffix
    to pin a specific inference provider; otherwise the router auto-selects
    among providers enabled on this token.
    """
    if not settings.HF_ENABLED:
        return None

    model = settings.HF_GENERATION_MODEL
    if settings.HF_GENERATION_PROVIDER:
        model = f"{model}:{settings.HF_GENERATION_PROVIDER}"

    headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": settings.HF_GENERATION_MAX_TOKENS,
        "temperature": settings.HF_GENERATION_TEMPERATURE,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.HF_API_TIMEOUT_SECONDS) as client:
            response = await client.post(_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        return None

    message = choices[0].get("message") or {}
    text = message.get("content")
    if not text or not isinstance(text, str):
        return None

    text = text.strip()
    return text or None