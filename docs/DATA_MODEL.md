# DataGuard Data Model

Core entities are organization-scoped where they contain tenant data:

- `organizations`: tenant boundary.
- `users` / `user_roles`: identity and authorization membership.
- `analyses`: analysis metadata and redacted result payload.
- `findings`: first-class sensitive-data findings.
- `pia_records`: PIA workflow state and evidence payload.
- `remediation_items`: governed corrective actions.
- `audit_events`: append-only, hash-chained audit evidence.
- `security_events`: security telemetry.
- `api_keys` / `refresh_tokens`: credential material represented by hashes where applicable.

## Invariants

1. Tenant-scoped rows carry `organization_id`.
2. PostgreSQL RLS is used for tenant-scoped resources where configured.
3. Application queries also constrain tenant identity.
4. Raw PII must not be added to ordinary logs or finding evidence.
5. Audit events are append-only.

The schema is an implementation description, not a claim that every enterprise data-governance feature is complete.
