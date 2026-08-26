# DataGuard Québec Enterprise API

Phase 10 introduces the FastAPI application boundary. API handlers remain thin: authentication/tenant context is resolved by dependencies and detection/risk logic remains in domain services.

## Endpoints

- `GET /health/live` — process liveness; no authentication.
- `GET /health/ready` — readiness boundary; no authentication.
- `POST /api/v1/analyze` — authenticated tenant-scoped text analysis.

OpenAPI is available at `/docs` and `/redoc` in the application runtime.

## Authentication

The current adapter validates a signed JWT using `DATAGUARD_JWT_SECRET` and `DATAGUARD_JWT_ALGORITHM`. Tokens must contain `sub`, `org_id`, and `exp`. The architecture is intentionally compatible with replacement by an OIDC/JWKS validator without changing business services.

Production deployments should use an external identity provider, asymmetric signing/JWKS where appropriate, key rotation and secret management. A shared development secret must never be used for production.

## Tenant isolation

The authenticated `org_id` is the authoritative organization context exposed to application services. Client payloads cannot choose the organization. Persistent repositories introduced in later phases must require this context in every tenant-scoped query and mutation.

## Sensitive-data handling

The analysis endpoint never returns raw detected values. Responses contain a redacted representation plus type, offsets, confidence and detector provenance. The submitted text is processed in memory and is not persisted by this endpoint.

## CORS and browser security

The default API configuration allows no browser origins. Production browser origins must be explicitly configured from trusted deployment configuration; wildcard production CORS is prohibited.

## API contract safety

Pydantic request models reject unknown fields and constrain text size and numeric ranges. Authentication failures return generic errors rather than token-validation details.

## Not yet claimed

This phase does not claim a complete enterprise identity platform, persistent audit implementation, malware sandbox, distributed rate limiter, or full OIDC provider integration. Those capabilities require the corresponding infrastructure and later phases.
