# Privacy Impact Assessment (PIA) Workflow

DataGuard provides a structured PIA workflow to capture project/system context, personal information, purposes, sources, recipients, storage, retention, risks, safeguards and ownership.

## Lifecycle

`DRAFT → IN_REVIEW → REQUIRES_REMEDIATION → IN_REVIEW → APPROVED → ARCHIVED`

An assessment may also return from `IN_REVIEW` to `DRAFT` when additional preparation is needed. Invalid transitions are rejected by the domain workflow.

Every transition creates a versioned history entry containing the PIA ID, version, prior/current state, actor, UTC timestamp and reason. Persistent audit storage is integrated with the platform audit architecture in the later audit/evidence phase.

## Human approval

Approval is an organizational governance action. DataGuard does not make a legal determination. The organization remains responsible for validating applicability, risks, safeguards and final approval.

## Required assessment areas

1. Project/system
2. Personal information
3. Purposes
4. Data sources
5. Recipients
6. Storage locations
7. Retention
8. Risks
9. Safeguards
10. Owner
11. Remediation/approval status
