# Phase 14 — DevSecOps / Deployment Baseline

## Delivered

- Non-root production Docker image.
- Minimal Python 3.11 slim runtime.
- Container health check.
- Local PostgreSQL + Redis + API Compose stack.
- Required secret placeholders; no production secrets in repository.
- Explicit CORS configuration.
- Deployment guidance for TLS, secrets, backups, logging, metrics and network isolation.
- Québec/Canadian residency guidance without certifying any provider.
- Explicit production limitations instead of simulated infrastructure.

## Verification

The repository's CI quality workflow runs Ruff, mypy and pytest with coverage. GitHub status must be checked on the resulting commit before treating CI as green.

## Production acceptance gates

Before government deployment, validate image scanning, SBOM generation, dependency policy, signed artifacts, secret scanning, TLS configuration, OIDC/JWKS integration, database migration/rollback, backup restore, centralized observability, network egress policy, malware scanning for uploads, resource limits and disaster recovery in the target environment.
