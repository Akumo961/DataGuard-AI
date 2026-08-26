# DataGuard Québec — Threat Model

## Scope

This threat model covers the Phase 3 API/security boundary and identifies controls that must remain enforced as later phases add storage, document processing, ML and integrations.

## Assets

- Personal information and sensitive findings.
- Tenant metadata and governance records.
- Authentication credentials and access tokens.
- Compliance/PIA evidence.
- Audit records.
- ML models and configuration.

## Threats and controls

| Threat | Primary controls | Residual work |
|---|---|---|
| Malicious upload | size/MIME/filename validation, isolated-worker contract | malware scanning and sandboxing in Phase 9 |
| Prompt injection | deterministic PII path does not require an LLM; treat document text as untrusted | AI-specific controls when LLM features are added |
| Data exfiltration | RBAC, tenant context, restrictive CORS, SSRF checks, no raw-document requirement | storage/network policy and DLP controls |
| Unauthorized access | JWT validation, issuer/audience checks, centralized RBAC | persistent identity/session controls |
| Insider threat | least-privilege roles and audit contract | immutable audit/evidence implementation |
| Model manipulation | replaceable model boundary and provenance requirement | signed model artifacts and supply-chain controls |
| Supply-chain attack | pinned dependency ranges, CI scanning planned | lockfile/SBOM/signing and deployment policy |
| API abuse | request limits and distributed rate limiting | WAF/API gateway policy |
| Tenant isolation failure | explicit TenantContext and fail-closed policy | database row-level isolation in Phase 4 |
| Sensitive-data leakage | minimized raw storage, secure headers, no-store API responses | redaction/pseudonymization pipeline |

## Trust boundaries

1. Browser/enterprise client → API gateway/application.
2. API → application/domain services.
3. Application → databases/object storage.
4. Application → worker/ML/OCR processes.
5. Application → external connectors/identity providers.

Content crossing a trust boundary is untrusted unless authenticated and validated.

## Security principles

- Default deny.
- Explicit tenant context.
- Least privilege.
- Short-lived credentials.
- Minimize sensitive data retention.
- Separate deterministic detection from generative AI.
- Log security-relevant events without logging raw personal information by default.
- Fail closed when a security dependency required for production enforcement is unavailable.

## Abuse cases to test

- Token signed with the wrong algorithm.
- Expired/not-yet-valid token.
- Wrong issuer or audience.
- Unknown role or permission.
- Cross-tenant resource identifier.
- Oversized request/upload.
- Path traversal filename.
- Local/private SSRF target.
- Rate-limit exhaustion.
- Malformed JWT claims.

This document is a design artifact, not a penetration-test report or security certification.
