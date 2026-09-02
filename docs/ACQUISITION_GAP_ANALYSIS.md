# DataGuard — Bank-Grade Acquisition Technical Gap Analysis

**Baseline reviewed:** `main` at `9346da5ed987f79ea0b4131ed393247b8d75f18d` plus the hardening branch changes through the current head.

This is an evidence-based engineering assessment. Documentation is not treated as proof of implementation, and valuation is not guaranteed.

## Current implementation matrix

| Capability | Status | Evidence / gap | Acquisition importance |
|---|---|---|---|
| FastAPI versioned API | 🟢 | Versioned analysis, PIA, remediation, audit and findings routes exist | High |
| Authentication | 🟡 | JWT validation and development auth exist; production identity is delegated to OIDC primitives, not a complete provider lifecycle | Critical |
| Centralized RBAC | 🟢 | Permissions are centralized in `AuthorizationPolicy` and injected into routes | Critical |
| Tenant isolation | 🟢/🟡 | PostgreSQL RLS + tenant predicates + cross-tenant E2E check; broader database-backed coverage is still required | Critical |
| Secure uploads | 🟢/🟡 | Size, filename, magic/MIME and OOXML protections; parser errors are now normalized; malware scanning/isolation remains deployment work | Critical |
| PII detection | 🟢/🟡 | Regex/context plus optional multilingual NER; no independently validated production benchmark yet | High |
| Data classification | 🟢 | Deterministic `RuleBasedClassifier` produces explainable PUBLIC/INTERNAL/CONFIDENTIAL/RESTRICTED/HIGHLY_RESTRICTED results with model/version provenance | Critical |
| First-class findings | 🟢 | Tenant-scoped `findings` table, RLS, persistence from analysis, filtered/paginated API | Critical |
| Risk engine | 🟡 | Deterministic explainable scoring exists; enterprise policy dimensions and calibration evidence remain | High |
| Regulatory mapping | 🟡 | Versioned YAML controls and evaluator exist; applicability/evidence workflow remains basic and is not legal compliance | High |
| PIA | 🟡 | Persistent state machine exists; rich vendor/jurisdiction/approval/evidence workbench is incomplete | Critical |
| Remediation | 🟡 | Creation/persistence exists; complete lifecycle, evidence, deadlines and approval APIs remain | Critical |
| Audit evidence | 🟢/🟡 | Hash chain + append-only controls + verification endpoint; actor provenance now supports external identity subjects; request/IP provenance is still incomplete in some paths | Critical |
| Enterprise API | 🟡 | OpenAPI is automatic, but organization/policy/integration administration is incomplete | Critical |
| Connectors | 🔴 | No verified S3/Azure/SharePoint/Drive/database/SIEM/IAM/ticketing adapters | Critical |
| Async processing | 🟡 | Queue/job contract exists, but no verified production worker implementation | Critical |
| Frontend | 🟡 | Responsive dashboard exists, but several enterprise areas remain placeholders/session-local | High |
| Observability | 🟡 | Request IDs, rate limiting and logging primitives exist; metrics/traces/alerts/job telemetry need expansion | High |
| CI/CD | ⚠️ | Quality workflow runs are currently marked failed with no retrievable jobs/check-runs through the available GitHub API surface; SBOM workflow has previously succeeded | Critical |
| Deployment | 🟡 | Docker/PostgreSQL/Redis foundation exists; Kubernetes/private-cloud/on-prem delivery artifacts are incomplete | High |
| Disaster recovery | 🟡 | Architecture/documentation exists; measured backup/restore and recovery exercises are still required | High |
| Documentation | 🟡 | Strong engineering documentation exists and must continue to track implementation exactly | High |

## Highest-priority blockers

### P0 — correctness/security

1. **CI execution evidence:** quality runs are marked failed with no retrievable jobs/check-runs through the available GitHub API surface. This prevents an honest green-CI claim and needs repository Actions investigation.
2. **Tenant isolation evidence:** RLS is materially implemented, but database-backed tests must cover read/write/delete/inference paths for every tenant-scoped resource.
3. **Production identity:** external OIDC/JWKS validation is implemented as a boundary, but provisioning, role mapping, key rotation, session/revocation and provider-specific SSO acceptance tests are not complete.
4. **Audit provenance:** actor identity is now represented as an external subject string; request ID and client provenance should be propagated consistently into every auditable action.
5. **Upload isolation:** parser hardening is improved, but production malware scanning, sandboxing, quarantine and controlled temporary storage still require deployment implementation.

### P1 — enterprise product depth

1. Organization-configurable classification policies and policy administration.
2. Findings lifecycle: review, suppress/confirm, assign, deadline, remediation link, evidence and approval.
3. Complete PIA model: processing purposes, data subjects, vendors, jurisdictions, safeguards, residual risk, reviewers and approvals.
4. Remediation lifecycle with SLA/deadline, ownership, evidence and audit history.
5. Versioned organization/policy/audit/finding APIs with consistent pagination, filtering, idempotency and correlation IDs.
6. Concrete asynchronous worker/queue implementation with retries, idempotency and backpressure.
7. Connector SDK plus at least one verified enterprise storage adapter.
8. Production metrics, traces, alerts and security-event operations.

### P2 — acquisition evidence

1. Reproducible PII/NER benchmark dataset, precision/recall/F1 and per-class error analysis.
2. Independent penetration test and remediation evidence.
3. Security/privacy control evidence mapped to recognized assurance frameworks.
4. Tested backup/restore and disaster-recovery exercises with measured RPO/RTO.
5. Real enterprise customers, retention, expansion and referenceable outcomes.
6. Demonstrated integration value and measurable reduction in privacy/security workload.
7. Verified ownership/licensing/IP provenance for source code, models, datasets and dependencies.

## Strategic acquisition wedge

The strongest technical wedge is **sensitive-data discovery + explainable classification/risk + regulatory mapping + governed remediation + tamper-evident evidence**. DataGuard should not try to win by matching the full breadth of mature privacy suites immediately.

## Prohibited claims until evidence exists

- SOC 2, ISO 27001, PCI, HIPAA, GDPR/CCPA/Québec legal compliance certification.
- Government accreditation or regulatory approval.
- Production detection accuracy percentages without reproducible benchmark evidence.
- Independent penetration-test claims without an actual report.
- Bank integrations without working adapters and integration tests.
- $9M–$15M valuation as a guaranteed outcome.

## Execution order

1. Restore and prove a genuinely green CI/E2E baseline.
2. Expand database-backed tenant-isolation and RBAC regression coverage.
3. Complete audit request/actor provenance.
4. Finish findings and remediation lifecycle APIs.
5. Make classification policy configurable per organization.
6. Expand the PIA evidence/approval model.
7. Implement asynchronous workers and durable job semantics.
8. Build and test the first enterprise connector.
9. Add operational observability and tested DR.
10. Build the acquisition evidence package from real customer/security/operational artifacts only.
