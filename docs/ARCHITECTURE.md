# DataGuard Québec — Target Architecture

## Purpose

Phase 2 establishes a modular boundary around the existing prototype. It does not claim that the platform is production-ready; the implementation will be completed phase by phase with evidence and tests.

## Logical architecture

```text
Enterprise Web UI
      |
      v
FastAPI API / authentication boundary
      |
      +--> application services
      |      +--> discovery/ingestion
      |      +--> document processing
      |      +--> PII detection pipeline
      |      +--> transparent risk engine
      |      +--> versioned compliance controls
      |      +--> PIA/remediation workflows
      |      +--> evidence/reporting
      |
      +--> PostgreSQL (tenant/governance metadata)
      +--> object storage (controlled raw documents, if retention permits)
      +--> Redis (bounded ephemeral jobs/rate limiting/cache)
      +--> workers (CPU/OCR/ML isolation)
      |
      +--> immutable-style audit sink
```

## Module boundaries

- `dataguard/backend`: transport/API only.
- `dataguard/core`: configuration and cross-cutting primitives.
- `dataguard/domain`: framework-independent domain contracts.
- `dataguard/ingestion`: source connectors and document acquisition.
- `dataguard/detection`: composable PII detectors and ensemble logic.
- `dataguard/classification`: sensitivity taxonomy and model inference.
- `dataguard/risk`: explainable scoring.
- `dataguard/compliance`: versioned control evaluation.
- `dataguard/audit`: audit/evidence persistence and export.
- `dataguard/security`: identity, authorization, upload/egress controls.
- `dataguard/database`: SQLAlchemy models, sessions and migrations.
- `dataguard/workers`: asynchronous processing jobs.

The API must depend on application/domain contracts rather than importing model internals directly. This keeps ML replaceable and makes security/policy decisions testable.

## Data lifecycle

1. Authenticate and authorize the tenant/user.
2. Validate upload/source metadata before accepting content.
3. Store only what the configured retention policy permits.
4. Extract text in an isolated worker.
5. Run layered detection and preserve provenance/confidence for each finding.
6. Compute risk from explicit factors; AI output is advisory.
7. Evaluate applicable controls and attach evidence references.
8. Persist governance metadata and audit events.
9. Redact/pseudonymize output where possible.
10. Delete raw content automatically when its retention period expires.

## Architectural non-goals

- No automatic legal-compliance determination.
- No fabricated ML metrics.
- No mandatory LLM dependency for deterministic PII discovery.
- No storage of raw sensitive content merely for convenience.
- No wildcard production CORS.

## Migration strategy

The existing `DataGuardAI/` prototype remains intact during the migration. Each phase moves one responsibility behind a tested interface. Legacy paths are removed only after equivalent behavior is covered by regression tests.
