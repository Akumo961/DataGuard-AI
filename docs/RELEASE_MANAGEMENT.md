# Release Management

## Pre-release

1. Freeze scope and identify release owner.
2. Run formatting, lint, type checks and the complete test suite.
3. Review dependency and container vulnerabilities.
4. Generate/retain SBOM and artifact provenance.
5. Review database migrations and rollback procedure.
6. Confirm configuration contains no committed secrets.
7. Review security/privacy/compliance-content changes.

## Promotion

Promote the same immutable artifact from staging to production. Do not rebuild an unverified production variant. Record commit SHA, artifact digest, configuration version and migration version.

## Post-release

Verify liveness/readiness, key API journeys, authentication, tenant boundaries, queues/workers, database connectivity, monitoring and alerting. Record acceptance evidence.

## Rollback

Stop promotion, preserve evidence, revert to the last known-good artifact and execute the approved database recovery procedure where necessary. Investigate before retrying.
