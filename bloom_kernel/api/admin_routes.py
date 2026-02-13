"""Admin routes — operations management portal.

All routes require admin authentication via require_admin().
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr

from bloom_kernel.services.mentor_service import MentorService
from bloom_kernel.utils.auth_deps import require_admin
from bloom_kernel.utils.database import db
from bloom_kernel.utils.response import fail, ok

router = APIRouter(prefix="/admin", tags=["admin"])

_mentor_svc = MentorService()


# ── Request Models ───────────────────────────────────────────────────

class CreateInvitationRequest(BaseModel):
    email: EmailStr
    mentor_id: str | None = None


class CreateMentorRequest(BaseModel):
    name: str
    bio: str = ""
    avatar_url: str | None = None
    persona_prompt: str = ""
    greeting_text: str | None = None
    style_tags: list[str] = []
    sort_order: int = 0
    status: str = "active"


# ── Invitation Management ────────────────────────────────────────────

@router.post("/invitations")
async def create_invitation(
    req: CreateInvitationRequest,
    admin_id: str = Depends(require_admin),
):
    """Create a mentor invitation (sends invite code)."""
    from bloom_kernel.config import settings
    from bloom_kernel.services.user_service import UserService

    svc = UserService(settings)
    async with db.session() as session:
        result = await svc.create_invitation(
            session, req.email, admin_id, req.mentor_id
        )
        return ok(result)


@router.get("/invitations")
async def list_invitations(admin_id: str = Depends(require_admin)):
    """List all mentor invitations."""
    from bloom_kernel.config import settings
    from bloom_kernel.services.user_service import UserService

    svc = UserService(settings)
    async with db.session() as session:
        invitations = await svc.list_invitations(session)
        return ok(invitations)


# ── Mentor Management (admin CRUD) ──────────────────────────────────

@router.get("/mentors")
async def list_all_mentors(admin_id: str = Depends(require_admin)):
    """List all mentors (including inactive, for admin overview)."""
    async with db.session() as session:
        mentors = await _mentor_svc.list_all(session)
        return ok(mentors)


@router.post("/mentors")
async def create_mentor(
    req: CreateMentorRequest,
    admin_id: str = Depends(require_admin),
):
    """Create a new mentor profile (admin only)."""
    async with db.session() as session:
        mentor_id = await _mentor_svc.create(session, req.model_dump())
        return ok({"id": mentor_id})


@router.delete("/mentors/{mentor_id}")
async def delete_mentor(mentor_id: str, admin_id: str = Depends(require_admin)):
    """Delete a mentor (admin only). Soft-delete by setting status='inactive'."""
    async with db.session() as session:
        updated = await _mentor_svc.update_mentor(session, mentor_id, {"status": "inactive"})
        if updated is None:
            return fail(404, "Mentor not found", status_code=404)
        return ok({"deleted": mentor_id})


# ── Clone Pipeline Management (admin) ────────────────────────────────

@router.post("/mentors/{mentor_id}/clone/cancel")
async def admin_cancel_clone(mentor_id: str, admin_id: str = Depends(require_admin)):
    """Cancel a mentor's active clone training (admin override)."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    result = await clone_svc.cancel_job(mentor_id)
    if result is None:
        return fail(404, "No active training job to cancel", status_code=404)
    return ok(result)


@router.post("/mentors/{mentor_id}/clone/resume")
async def admin_resume_clone(mentor_id: str, admin_id: str = Depends(require_admin)):
    """Resume a mentor's clone training from last checkpoint (admin override)."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    try:
        job = await clone_svc.start_clone(mentor_id, resume=True)
        return ok(job)
    except ValueError as e:
        return fail(400, str(e), status_code=400)
    except Exception as e:
        return fail(500, f"Clone resume failed: {e}", status_code=500)


# ── Safety Incident Management ──────────────────────────────────────

@router.get("/safety/incidents")
async def list_incidents(
    session_id: str | None = None,
    unresolved_only: bool = False,
    admin_id: str = Depends(require_admin),
):
    """List safety incidents (admin view)."""
    from bloom_kernel.services.safety_service import SafetyService

    safety_svc = SafetyService()
    async with db.session() as session:
        incidents = await safety_svc.list_incidents(
            session, session_id=session_id, unresolved_only=unresolved_only
        )
        return ok(incidents)


@router.post("/safety/incidents/{incident_id}/resolve")
async def resolve_incident(
    incident_id: str,
    admin_id: str = Depends(require_admin),
):
    """Mark a safety incident as resolved."""
    from bloom_kernel.services.safety_service import SafetyService

    safety_svc = SafetyService()
    async with db.session() as session:
        result = await safety_svc.resolve_incident(session, incident_id, admin_id)
        if result is None:
            return fail(404, "Incident not found", status_code=404)
        return ok(result)
