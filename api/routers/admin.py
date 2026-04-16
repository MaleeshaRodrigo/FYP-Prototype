"""GET /api/admin/usage, /api/admin/audit — Hospital admin endpoints."""

import random

from fastapi import APIRouter, Depends, Query

from ..core.security import require_role
from ..domain.schemas import AuditLogEntry, UsageStatsResponse

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/usage", response_model=UsageStatsResponse)
async def get_usage_stats(
    period: str = Query(default="daily", pattern="^(daily|weekly|monthly)$"),
    _user: dict = Depends(require_role("admin")),
) -> UsageStatsResponse:
    """Returns scan volume and detection rate statistics."""
    daily = [
        {"date": f"2025-04-{str(i).zfill(2)}", "scans": 60 + random.randint(0, 40), "mel": random.randint(0, 12)}
        for i in range(1, 16)
    ]
    return UsageStatsResponse(
        total_scans=1247,
        mel_detections=89,
        mel_rate=0.0713,
        referral_rate=0.124,
        avg_inference_ms=138,
        period_start="2025-04-01T00:00:00Z",
        period_end="2025-04-15T23:59:59Z",
        daily_counts=daily,
    )


@router.get("/audit", response_model=list[AuditLogEntry])
async def get_audit_log(
    action: str | None = Query(default=None),
    _user: dict = Depends(require_role("admin")),
) -> list[AuditLogEntry]:
    """Returns the platform audit log, optionally filtered by action."""
    entries = [
        AuditLogEntry(id="a001", action="model:activated", user="admin@hare.med", target="v8", timestamp="2025-04-10T09:15:00Z", details="Activated v8-PGD-AT"),
        AuditLogEntry(id="a002", action="parameters:updated", user="admin@hare.med", target="ga-params", timestamp="2025-04-10T09:20:00Z", details="θ=0.3985, τ=0.7671, α=0.5467"),
        AuditLogEntry(id="a003", action="scan:completed", user="clinician@hare.med", target="img_001", timestamp="2025-04-11T14:30:00Z", details="MEL detected, confidence 0.72"),
        AuditLogEntry(id="a004", action="model:deprecated", user="admin@hare.med", target="v7", timestamp="2025-04-12T08:00:00Z", details="Deprecated v7 in favour of v8"),
        AuditLogEntry(id="a005", action="scan:completed", user="clinician@hare.med", target="img_002", timestamp="2025-04-13T11:15:00Z", details="NON_MEL, confidence 0.15"),
    ]
    if action:
        entries = [e for e in entries if e.action == action]
    return entries
