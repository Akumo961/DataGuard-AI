# Architecture

## Status

The repository now contains the Phase 2 architecture foundation plus implemented Phase 3–14 components. Interfaces remain explicit where a production adapter has not yet been delivered; an interface is never treated as proof of implementation.

## Logical architecture

```text
Enterprise Web UI
      |
      v
FastAPI transport/API
      |
      v
Application use cases
      +--> ingestion / document processing
      +--> layered PII detection / classification
      +--> explainable risk engine
      +--> versioned compliance controls
      +--> PIA workflow
      +--> audit/evidence
      +--> persistence / tenant scope
      +--> security policy
      +--> background jobs
```

## Boundaries

- `dataguard/backend`: HTTP transport and dependency wiring.
- `dataguard/application`: use-case orchestration.
- `dataguard/core`: configuration and cross-cutting primitives.
- `dataguard/domain`: framework-independent domain models.
- `dataguard/ingestion`: source connector contracts.
- `dataguard/processing`: bounded document processing.
- `dataguard/detection`: composable PII detection.
- `dataguard/classification`: sensitivity inference.
- `dataguard/risk`: transparent risk scoring.
- `dataguard/compliance`: versioned control evaluation.
- `dataguard/pia`: PIA state machine and history model.
- `dataguard/audit`: audit/evidence models and integrity service.
- `dataguard/security`: identity/authorization boundaries.
- `dataguard/database`: tenant-scoped persistence boundaries.

## Security and privacy principles

Authentication context supplies tenant identity; clients must not be trusted to choose an organization. Sensitive output is redacted where possible. Raw documents should be minimized and processed in isolated workers in production. Compliance and AI outputs are advisory and require human governance/legal review.

## Deployment

The repository includes a non-root container and a local PostgreSQL/Redis/API Compose stack. Production deployment additionally requires organization-controlled TLS, secrets management, observability, backup/restore, artifact/image security, OIDC/JWKS integration and isolated processing infrastructure.
