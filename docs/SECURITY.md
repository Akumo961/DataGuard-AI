# DataGuard Security

## Scope

DataGuard treats customer content and derived sensitive-data findings as confidential. This document describes implemented controls and explicitly identifies controls that still require deployment or independent assurance.

## Implemented controls

- JWT validation at the API boundary with issuer/audience checks when configured.
- Locally issued access tokens have unique `jti` identifiers and can be revoked through Redis until expiry.
- OIDC/JWKS validation reuses a bounded PyJWKClient cache and refreshes on unknown signing keys, supporting normal provider key rotation behavior.
- Centralized permission policy with fail-closed authorization.
- Organization-scoped PostgreSQL RLS for tenant-scoped tables.
- Tenant predicates in application repositories/routes.
- Organization-configurable classification policy with explicit RBAC and audit logging.
- Request size limits and rate limiting middleware.
- Sanitized request ID, client IP and user-agent provenance is attached to audit events without trusting forwarded headers by default.
- Upload filename, extension, magic-byte, MIME and archive/OOXML checks.
- Optional ClamAV INSTREAM malware scanning before document extraction; production configuration now fails closed when no scanner endpoint is supplied.
- Same-origin restrictive CSP and security response headers.
- Redacted detection output; raw detected values are not persisted by the findings model.
- Append-only audit events with organization-scoped hash chaining.
- Secrets supplied through environment/configuration rather than frontend code.
- S3-compatible connector enforces a tenant prefix and is documented for read/list least-privilege IAM.

## Still required before enterprise assurance

- Independent penetration test.
- Production OIDC provider acceptance tests, provisioning, claim-to-role mapping and lifecycle controls.
- Managed secret/KMS deployment evidence.
- Malware quarantine/sandboxing and an operational response process around scanner detections.
- Formal vulnerability-management SLA and SBOM publication process.
- External assurance such as SOC 2/ISO 27001 only after the applicable audit process is actually completed.
- Multi-replica distributed tracing/alert delivery evidence.

No certification or regulatory compliance claim is made by this document.
