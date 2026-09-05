"""
Routing / decision policy (Theme 1 & 2).

Turns a confidence score + context into one of four decisions:

- ``clarification_requested`` - the request itself is too ambiguous to
  classify reliably; ask the user a clarifying question rather than guess
  (checked by the caller *before* this policy, see app/services/agent.py).
- ``auto_send``            - confidence is high enough to answer directly.
- ``pending_validation``   - answer is plausible but not certain enough to
  send unsupervised; a staff member must approve it (internal validation).
- ``escalated``            - confidence is too low, or urgency is high, or
  there is no grounded source at all; hand off to a human support agent.
"""

from dataclasses import dataclass

from app.core.config import settings


@dataclass
class RoutingDecision:
    decision: str
    priority: str


def decide(confidence_score: float, urgency: str, has_grounded_source: bool) -> RoutingDecision:
    # Urgent requests always get at least human review, even if the model is
    # confident, because the cost of a wrong "auto" answer under urgency is
    # higher than the cost of a short delay.
    if urgency == "urgent":
        return RoutingDecision(decision="escalated", priority="urgent")

    if not has_grounded_source:
        priority = "high" if urgency == "high" else "normal"
        return RoutingDecision(decision="escalated", priority=priority)

    if confidence_score >= settings.CONFIDENCE_AUTO_THRESHOLD:
        return RoutingDecision(decision="auto_send", priority="normal")

    if confidence_score >= settings.CONFIDENCE_VALIDATION_THRESHOLD:
        priority = "high" if urgency == "high" else "normal"
        return RoutingDecision(decision="pending_validation", priority=priority)

    priority = "high" if urgency == "high" else "normal"
    return RoutingDecision(decision="escalated", priority=priority)
