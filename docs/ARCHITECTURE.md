# DataGuard Québec — Target Architecture

## Phase 2 status

**Architecture foundation complete.** This phase establishes enforceable module boundaries and application/domain contracts without pretending that security, persistence, ML, compliance, or enterprise UI are already implemented. Those capabilities remain explicit later-phase deliverables.

## Logical architecture

```text
Enterprise Web UI
      |
      v
FastAPI transport/API
      |
      v
Application use cases
      |
      +--> ingestion connectors
      +--> document processing
      +--> layered PII detection
      +--> sensitivity classification
      +--> transparent risk engine
      +--> versioned compliance engine
      +--> PIA/remediation workflows
      +--> audit/evidence
      |
      +--> persistence ports --> PostgreSQL adapter
      +--> object storage port --> controlled object store
      +--> job port --> Redis/broker adapter
      |
      +--> security ports --> OIDC/RBAC implementation
```

## Module boundaries

- `dataguard/backend`: HTTP transport and dependency wiring; no ML/business-rule internals.
- `dataguard/application`: use-case orchestration and transaction boundaries.
- `dataguard/core`: configuration and cross-cutting primitives.
- `dataguard/domain`: framework-independent domain models.
- `dataguard/ingestion`: read-only source connector contracts.
- `dataguard/processing`: bounded document extraction contracts.
- `dataguard/detection`: composable PII detector contracts.
- `dataguard/classification`: sensitivity/model inference contract.
- `dataguard/risk`: explainable risk scoring contract.
- `dataguard/compliance`: versioned control evaluation contract.
- `dataguard/audit`: audit/evidence sink contract.
- `dataguard/security`: identity and authorization ports; implementation is Phase 3.
- `dataguard/database`: tenant-scoped persistence ports; concrete SQLAlchemy models/migrations are Phase 4.
- `dataguard/workers`: asynchronous job contract; broker implementation follows when required.

## Dependency rules

1. Domain models do not import FastAPI, SQLAlchemy, ML libraries, or infrastructure.
2. Application services depend on domain contracts/ports, never concrete ML or database adapters.
3. Adapters implement ports and may depend inward, never the reverse.
4. API routes perform transport validation and invoke application use cases.
5. Every persistence port accepts an explicit `TenantContext`; adapters must enforce tenant scope at query boundaries.
6. Security authorization is centralized behind a policy port rather than duplicated in use cases.
7. ML components are replaceable and must expose provenance/confidence; deterministic discovery must not depend on an LLM.

## Data lifecycle

1. Authenticate and authorize the tenant/user.
2. Validate source metadata and content before accepting it.
3. Store only what the configured retention policy permits.
4. Extract text in an isolated/bounded worker.
5. Run layered detection and preserve provenance/confidence for each finding.
6. Compute risk from explicit factors; AI output is advisory.
7. Evaluate applicable controls and attach evidence references.
8. Persist governance metadata and audit events.
9. Redact/pseudonymize output where possible.
10. Delete raw content automatically when its retention period expires.

## Contracts delivered in Phase 2

- `AnalysisResult`, `Detection`, `RiskAssessment`, `PIIType`, and `RiskLevel` domain types.
- `Connector` ingestion port.
- `DocumentProcessor` extraction port.
- `DetectionEngine` detector port.
- `Classifier` sensitivity classification port.
- `RiskEngine` scoring port.
- `ComplianceEngine` control-evaluation port.
- `AnalysisRepository` and `UnitOfWork` persistence ports.
- `IdentityProvider` and `AuthorizationPolicy` security ports.
- `JobQueue` worker port.
- `AnalyzeText` application orchestration use case.
- Contract tests proving detector/risk/compliance components can be substituted without changing the use case.

## Architectural non-goals

- No automatic legal-compliance determination.
- No fabricated ML metrics.
- No mandatory LLM dependency for deterministic PII discovery.
- No storage of raw sensitive content merely for convenience.
- No wildcard production CORS.
- No claim that a port/interface is equivalent to a production implementation.

## Migration strategy

The existing `DataGuardAI/` prototype remains intact while responsibilities migrate behind the new contracts. Legacy paths are removed only after equivalent behavior is covered by regression tests. Phase 3 implements the security foundation; Phase 4 implements persistence and tenant isolation; later phases attach production adapters.
