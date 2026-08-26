# DataGuard Québec — Security Architecture

## Phase 3 status

Phase 3 establishes the security foundation. It is not a claim of certification, accreditation, or complete government deployment readiness.

## Security controls implemented

- Argon2id password hashing through `pwdlib`.
- Short-lived JWT access-token validation.
- OIDC/JWKS-ready asymmetric token validation for RS256/ES256.
- Explicit issuer/audience validation when configured.
- Centralized RBAC policy with fail-closed unknown permissions.
- Explicit organization/tenant context and cross-tenant denial.
- Strict CORS validation; wildcard origins are rejected.
- Trusted-host validation.
- Request-ID propagation.
- Security response headers and API no-store cache policy.
- Request body size limits.
- Upload filename, size and MIME allow-list validation.
- SSRF-oriented outbound URL checks against local/private/link-local/reserved destinations.
- Redis-backed distributed rate limiting; local in-memory limiting is only a development/test fallback.
- Production API documentation is disabled by default.

## Authentication model

DataGuard does not implement a bespoke identity provider. The intended government deployment integrates an approved enterprise identity provider using OIDC. The API validates signed access tokens and maps claims to a tenant and roles.

Local development can use explicitly configured HS256 tokens. Production configuration rejects HS256 and requires OIDC/JWKS configuration.

## Authorization model

Authorization is centralized in `AuthorizationPolicy`. Every resource operation must supply the organization identifier of the resource and the authenticated `TenantContext`. A mismatch is denied before role evaluation. This is a foundation for database-level tenant isolation in Phase 4; application-level checks alone are not considered sufficient for the final architecture.

## Upload security

File metadata is validated before ingestion. MIME allow-lists and size limits reduce attack surface, while extraction must run in an isolated worker. Phase 9 will add bounded parsers, OCR isolation and malware scanning integration. The current upload helpers do not constitute antivirus protection.

## Secrets

Secrets are represented by typed settings and must come from environment variables or a managed secret store. No credentials are committed. Production asymmetric OIDC keys are obtained from the configured JWKS endpoint rather than stored in application source.

## Transport

TLS termination should occur at an approved ingress/load balancer with HTTP-to-HTTPS redirection and modern TLS policy. HSTS is emitted when the application receives HTTPS traffic. Deployment infrastructure must ensure the application cannot be reached around the trusted TLS boundary.

## Security limitations remaining

- Persistent users, sessions, refresh-token rotation and account lockout require the Phase 4 identity persistence layer.
- Database row-level tenant isolation requires Phase 4.
- Malware scanning and sandboxed document extraction require Phase 9.
- Central security-event storage and immutable audit evidence require Phase 12.
- Production OIDC issuer/JWKS endpoints must be supplied and validated by deployment security engineering.

These are deliberately documented as remaining work rather than simulated functionality.
