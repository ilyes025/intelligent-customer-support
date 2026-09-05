"""
Aggregated production metrics for the staff dashboard.

Pulls live numbers from AgentDecisionLog / SupportTicket (what the agent has
actually decided in production) and, separately, the most recent offline
eval-harness run (eval/results/latest.json, produced by eval/run_eval.py) -
so the dashboard shows both "how is it behaving live" and "how did it score
against the golden set last time we checked."

Both halves degrade gracefully: an empty decision log returns zeroed stats
instead of dividing by zero, and a missing eval results file simply omits
that section rather than erroring.
"""

import json
from pathlib import Path
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.support import AgentDecisionLog, SupportTicket

EVAL_RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "results" / "latest.json"


async def get_decision_distribution(db: AsyncSession) -> dict:
    stmt = select(AgentDecisionLog.decision, func.count()).group_by(AgentDecisionLog.decision)
    result = await db.execute(stmt)
    return {decision: count for decision, count in result.all()}


async def get_avg_confidence_by_intent(db: AsyncSession) -> dict:
    stmt = (
        select(AgentDecisionLog.intent, func.avg(AgentDecisionLog.confidence_score))
        .filter(AgentDecisionLog.intent.is_not(None))
        .group_by(AgentDecisionLog.intent)
    )
    result = await db.execute(stmt)
    return {intent: round(avg, 3) for intent, avg in result.all() if avg is not None}


async def get_avg_faithfulness(db: AsyncSession) -> Optional[float]:
    stmt = select(func.avg(AgentDecisionLog.faithfulness_score)).filter(
        AgentDecisionLog.faithfulness_score.is_not(None)
    )
    result = await db.execute(stmt)
    avg = result.scalar_one_or_none()
    return round(avg, 3) if avg is not None else None


async def get_faithfulness_over_time(db: AsyncSession, limit: int = 100) -> list:
    stmt = (
        select(AgentDecisionLog.created_at, AgentDecisionLog.faithfulness_score)
        .filter(AgentDecisionLog.faithfulness_score.is_not(None))
        .order_by(AgentDecisionLog.created_at.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [
        {"timestamp": ts.isoformat(), "faithfulness_score": score}
        for ts, score in result.all()
    ]


async def get_ticket_stats(db: AsyncSession) -> dict:
    stmt = select(SupportTicket.status, func.count()).group_by(SupportTicket.status)
    result = await db.execute(stmt)
    by_status = {status: count for status, count in result.all()}

    total_stmt = select(func.count()).select_from(SupportTicket)
    total = (await db.execute(total_stmt)).scalar_one()

    resolved_stmt = select(func.count()).select_from(SupportTicket).filter(
        SupportTicket.status == "resolved"
    )
    resolved = (await db.execute(resolved_stmt)).scalar_one()

    return {
        "by_status": by_status,
        "total": total,
        "resolution_rate": round(resolved / total, 3) if total else None,
    }


def get_latest_eval_results() -> Optional[dict]:
    """Read the most recent eval/run_eval.py output, or None if it has
    never been run. Never raises - a missing/corrupt file just means the
    dashboard's "offline eval" section is omitted."""
    if not EVAL_RESULTS_PATH.exists():
        return None
    try:
        return json.loads(EVAL_RESULTS_PATH.read_text())
    except (ValueError, OSError):
        return None


async def build_metrics_summary(db: AsyncSession) -> dict:
    decision_distribution = await get_decision_distribution(db)
    total_decisions = sum(decision_distribution.values())
    auto_send_rate = (
        round(decision_distribution.get("auto_send", 0) / total_decisions, 3)
        if total_decisions
        else None
    )
    escalation_rate = (
        round(decision_distribution.get("escalated", 0) / total_decisions, 3)
        if total_decisions
        else None
    )

    return {
        "live": {
            "total_decisions": total_decisions,
            "decision_distribution": decision_distribution,
            "auto_send_rate": auto_send_rate,
            "escalation_rate": escalation_rate,
            "avg_confidence_by_intent": await get_avg_confidence_by_intent(db),
            "avg_faithfulness": await get_avg_faithfulness(db),
            "faithfulness_over_time": await get_faithfulness_over_time(db),
            "tickets": await get_ticket_stats(db),
        },
        "eval": get_latest_eval_results(),
    }
