# DataGuard Security

## Scope

DataGuard treats customer content and derived sensitive-data findings as confidential. This document describes implemented controls and explicitly identifies controls that still require deployment or independent assurance.

## Implemented controls

- JWT validation at the API boundary.
- Centralized permission policy with fail-closed authorization.
- Organization-scoped PostgreSQL RLS for tenant-scoped tables.
- Tenant predicates in application repositories/routes.
- Request size limits and rate limiting middleware.
- Upload filename, extension, magic-byte, MIME and archive/OOXML checks.
- Same-origin restrictive CSP and security response headers.
- Redacted detection output; raw detected values are not persisted by the findings model.
- Append-only audit events with organization-scoped hash chaining.
- Secrets supplied through environment/configuration rather than frontend code.

## Still required before enterprise assurance

- Independent penetration test.
- Production OIDC provider acceptance tests, provisioning and lifecycle controls.
- Managed secret/KMS deployment evidence.
- Malware scanning/quarantine/sandboxing for uploaded content.
- Formal vulnerability-management SLA and SBOM publication process.
- External assurance such as SOC 2/ISO 27001 only after the applicable audit process is actually completed.

No certification or regulatory compliance claim is made by this document.
