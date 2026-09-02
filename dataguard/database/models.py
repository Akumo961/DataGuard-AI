from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Uuid,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from dataguard.database.base import Base, TenantScopedMixin, TimestampMixin, UUIDPrimaryKeyMixin
from dataguard.security.audit_context import get_context


class Organization(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "organizations"
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    classification_policy: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    users: Mapped[list[User]] = relationship(back_populates="organization")


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    external_subject: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(512), nullable=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    failed_login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    roles: Mapped[list[UserRole]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    organization: Mapped[Organization] = relationship(back_populates="users")
    __table_args__ = (Index("uq_users_org_email", "organization_id", "email", unique=True),)


class UserRole(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "user_roles"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    user: Mapped[User] = relationship(back_populates="roles")
    __table_args__ = (Index("uq_user_roles", "organization_id", "user_id", "role", unique=True),)


class APIKey(UUIDPrimaryKeyMixin, TimestampMixin, TenantScopedMixin, Base):
    __tablename__ = "api_keys"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(32), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Analysis(UUIDPrimaryKeyMixin, TimestampMixin, TenantScopedMixin, Base):
    __tablename__ = "analyses"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="COMPLETED")
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    result: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (Index("uq_analysis_org_id", "organization_id", "id", unique=True),)


class Finding(UUIDPrimaryKeyMixin, TimestampMixin, TenantScopedMixin, Base):
    __tablename__ = "findings"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_id: Mapped[UUID] = mapped_column(nullable=False, index=True)
    pii_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    detector: Mapped[str] = mapped_column(String(80), nullable=False)
    classification_label: Mapped[str] = mapped_column(String(40), nullable=False)
    classification_confidence: Mapped[float] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="OPEN")
    owner_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        ForeignKeyConstraint(
            ["analysis_id", "organization_id"],
            ["analyses.id", "analyses.organization_id"],
            ondelete="CASCADE",
            name="fk_findings_analysis_tenant",
        ),
    )


class PIARecord(UUIDPrimaryKeyMixin, TimestampMixin, TenantScopedMixin, Base):
    __tablename__ = "pia_records"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="DRAFT")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    owner_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class RemediationItem(UUIDPrimaryKeyMixin, TimestampMixin, TenantScopedMixin, Base):
    __tablename__ = "remediation_items"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("analyses.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(String(4000), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="OPEN")
    priority: Mapped[str] = mapped_column(String(32), nullable=False, default="MEDIUM")
    owner_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sla_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=168)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


@event.listens_for(RemediationItem, "before_insert")
def _set_remediation_due_at(mapper, connection, target: RemediationItem) -> None:
    del mapper, connection
    if target.due_at is None:
        target.due_at = datetime.now(timezone.utc) + timedelta(hours=target.sla_hours)


class SecurityEvent(UUIDPrimaryKeyMixin, TenantScopedMixin, Base):
    __tablename__ = "security_events"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AuditEvent(UUIDPrimaryKeyMixin, TenantScopedMixin, Base):
    __tablename__ = "audit_events"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    actor_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    object_type: Mapped[str] = mapped_column(String(100), nullable=False)
    object_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    previous_state: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    new_state: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    client: Mapped[str | None] = mapped_column(String(255), nullable=True)
    result: Mapped[str] = mapped_column(String(32), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    integrity_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")


@event.listens_for(AuditEvent, "before_insert")
def _hash_audit_event(mapper, connection, target: AuditEvent) -> None:
    del mapper
    if target.id is None:
        target.id = uuid4()
    context = get_context()
    target.request_id = target.request_id or context.request_id
    target.ip_address = target.ip_address or context.ip_address
    target.client = target.client or context.client
    org = str(target.organization_id)
    if connection.dialect.name == "postgresql":
        connection.execute(text("SELECT pg_advisory_xact_lock(hashtext(:org))"), {"org": org})
    previous = connection.execute(
        text(
            "SELECT integrity_hash FROM audit_events "
            "WHERE organization_id = :org "
            "ORDER BY occurred_at DESC, id DESC LIMIT 1"
        ),
        {"org": org},
    ).scalar()
    previous_hash = str(previous or "")
    occurred = target.occurred_at
    payload = {
        "event_id": str(target.id),
        "timestamp": occurred.isoformat() if occurred else "",
        "user_id": str(target.actor_id) if target.actor_id else "",
        "organization_id": org,
        "action": target.action,
        "object_type": target.object_type,
        "object_id": target.object_id or "",
        "previous_state": target.previous_state,
        "new_state": target.new_state,
        "ip_address": target.ip_address,
        "client": target.client,
        "request_id": target.request_id or "",
        "result": target.result,
        "previous_hash": previous_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    target.previous_hash = previous_hash
    target.integrity_hash = hashlib.sha256(canonical).hexdigest()


@event.listens_for(AuditEvent, "before_update")
def _reject_audit_update(mapper, connection, target: AuditEvent) -> None:
    raise RuntimeError("Audit events are append-only and cannot be updated")


@event.listens_for(AuditEvent, "before_delete")
def _reject_audit_delete(mapper, connection, target: AuditEvent) -> None:
    raise RuntimeError("Audit events are append-only and cannot be deleted")


class RefreshToken(UUIDPrimaryKeyMixin, TimestampMixin, TenantScopedMixin, Base):
    __tablename__ = "refresh_tokens"
    organization_id: Mapped[UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
