# Phase 3 — Security Foundation

## Status

**Complete as a security foundation.** This phase establishes reusable security enforcement and explicit production prerequisites. It does not claim penetration-test completion, certification, accreditation, or government authorization.

## Delivered

- Argon2id password hashing and verification.
- JWT access-token validation with required claims and bounded lifetime for locally issued development tokens.
- OIDC/JWKS validation path for RS256/ES256 production tokens.
- Explicit issuer and audience validation when configured.
- Centralized RBAC roles and permissions.
- Explicit tenant context with fail-closed cross-tenant authorization.
- Non-reversible API-key generation and verification primitives.
- Strict CORS and trusted-host configuration.
- Secure HTTP response headers and request IDs.
- API request body-size limits.
- Distributed Redis rate limiting for production; bounded in-memory fallback only outside production.
- Filename/path traversal and MIME/size upload validation.
- SSRF-oriented outbound destination validation.
- Production configuration rejects wildcard CORS and HTTP production origins and requires OIDC/JWKS configuration.
- CI quality workflow for linting, type checking and tests.
- Security architecture and threat model documentation.

## Explicitly not claimed

- No bespoke identity provider.
- No persistent user/session/account-lockout implementation; persistence is Phase 4.
- No database row-level security; tenant persistence is Phase 4.
- No antivirus/sandbox for documents; document security is completed with Phase 9 processing infrastructure.
- No immutable audit storage; evidence implementation is Phase 12.
- No security certification, government approval or legal compliance determination.

## Verification

The repository includes automated security tests for password hashing, JWT integrity, RBAC/tenant isolation, upload/SSRF defenses, API-key primitives and production configuration.

A local test execution was not possible in this environment because outbound GitHub/network DNS is unavailable to the execution container. The CI workflow is therefore the authoritative execution path once GitHub Actions schedules the branch commit. No test result is fabricated here.

## Exit criteria

Phase 3 is considered complete because all Phase 3 security responsibilities have concrete code, tests, configuration controls and documented residual dependencies. Remaining controls are intentionally assigned to their owning later phases rather than represented as completed security features.
