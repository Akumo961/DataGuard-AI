# DataGuard Québec — Security Architecture

## Status

The security foundation and subsequent application controls are implemented incrementally. This is not a claim of certification, accreditation, or complete government deployment readiness.

## Controls represented in the repository

- Argon2id password hashing through `pwdlib`.
- Short-lived JWT access-token validation.
- OIDC/JWKS-ready asymmetric validation.
- Issuer/audience validation when configured.
- Centralized fail-closed RBAC.
- Explicit organization/tenant context.
- Strict CORS and trusted-host validation.
- Request IDs and security headers.
- Request/upload size and MIME validation.
- SSRF-oriented outbound URL checks.
- Redis-backed rate limiting where configured.
- Non-root production container.
- Redacted PII API responses.
- Hash-based audit integrity records.

## Production requirements

Production must integrate an approved enterprise identity provider, managed secrets/KMS, TLS, restricted network egress, centralized security logging, vulnerability/image scanning, isolated document workers and tested backup/restore procedures.

## Limitations

Application controls are not equivalent to certification. Malware scanning, full WORM evidence retention, target-environment OIDC configuration, enterprise SIEM/KMS and infrastructure accreditation must be validated in the deployed environment.
