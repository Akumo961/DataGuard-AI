# DataGuard Québec

> **Enterprise privacy automation, PII discovery, risk assessment, and compliance workflow platform.**

DataGuard Québec is a production-oriented engineering foundation for discovering and classifying sensitive information, assessing privacy risk, supporting Privacy Impact Assessment (PIA) workflows, applying configurable compliance controls, and maintaining auditable evidence.

The project is designed around a clear principle: **privacy automation should be explainable, testable, tenant-aware, and reviewable by humans.**

> **Important:** DataGuard does not automatically establish legal compliance, certification, government approval, or regulatory conformity. Compliance controls and reports require organizational and qualified legal/privacy review.
>
> **Identity boundary:** production deployments delegate authentication to an external OIDC provider. The repository's `/api/v1/auth/register` and `/api/v1/auth/login` endpoints are intentionally limited to `DATAGUARD_ENVIRONMENT=development` for local testing only; they are not a production identity service.

## Why DataGuard?

Privacy teams often need to identify sensitive information across documents, understand the associated risk, map findings to applicable controls, and preserve evidence for review. DataGuard provides an engineering foundation for these workflows, with deterministic PII detection and explainable risk assessment as the currently implemented analysis path.

## Engineering highlights

- FastAPI application with versioned APIs
- Multi-layer PII detection architecture with deterministic pattern/context rules
- Optional lightweight multilingual spaCy NER for PERSON/LOCATION/ORGANIZATION detection
- Deterministic and explainable privacy risk scoring
- PostgreSQL / SQLAlchemy / Alembic persistence foundations
- PostgreSQL Row-Level Security (RLS) migration support for tenant isolation
- JWT authentication and tenant-aware authorization boundaries
- Development-only local registration/login; production identity delegated to OIDC
- PIA workflow and state-transition model
- Versioned Québec, Canadian, GDPR, and CCPA control definitions
- Document extraction, validation, and OCR interfaces
- Audit and evidence service foundation, including tenant-scoped integrity verification
- Responsive enterprise web dashboard
- Automated unit, API, security, and domain tests
- GitHub Actions quality workflow
- Docker-based local development
- Synthetic demonstration-data generator

## Architecture

```text
                         Enterprise Web UI
                                │
                                ▼
                         ┌──────────────┐
                         │   FastAPI    │
                         │  Versioned   │
                         │     API      │
                         └──────┬───────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
                 ▼              ▼              ▼
          Authentication    Analysis       Audit / Evidence
          + Tenant Context    Services          Services
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
        PII Detection       Risk Engine       PIA Workflow
              │                 │                  │
              ▼                 ▼                  ▼
        Findings / PII     Explainable Risk    State / Controls
              │                 │                  │
              └─────────────────┼──────────────────┘
                                ▼
                       PostgreSQL / SQLAlchemy
                                │
                         Tenant Isolation
                            + RLS
```

## Privacy intelligence workflow

```text
Document / Record
       │
       ▼
Extraction + Validation
       │
       ▼
PII / Sensitive Information Detection
       │
       ▼
Finding + Evidence
       │
       ▼
Explainable Risk Assessment
       │
       ▼
Compliance Control Mapping
       │
       ▼
PIA / Review Workflow
       │
       ▼
Audit Evidence
```

## Detection and AI/ML boundary

The current production analysis path is **deterministic plus an optional configured NER model**. The multilingual NER path uses the lightweight `xx_ent_wiki_sm` spaCy model when `DATAGUARD_NER_MODEL` is configured. Its checked-in synthetic benchmark reports precision/recall/F1 at runtime; the repository does not treat model-card metrics or synthetic results as a government-use accuracy guarantee.

The repository contains legacy/prototype AI/ML artifacts, but these are not treated as trained production models or validation evidence. No production accuracy claim is made from those artifacts.

A model-backed detector should only be presented as production AI after reproducible datasets, model provenance, measured precision/recall/F1, per-class error analysis, and human-review controls are available.

### No unsupported accuracy claims

The repository does **not** claim a production PII detection accuracy such as "95%+" because the current repository does not contain sufficient reproducible evidence to support that figure.

This is intentional: **measured performance is more valuable than an unverified benchmark claim.**

## Security & privacy

Security is treated as a system property rather than a prompt or model feature.

Current engineering controls include:

- JWT authentication boundary
- Tenant-aware authorization
- PostgreSQL RLS migration support
- Input validation through Pydantic contracts
- Audit/evidence foundations
- Environment-based configuration
- Security headers, request-size limits, trusted-host controls, and rate limiting
- Synthetic data for demonstrations
- Explicit production-data restrictions

Production deployments require appropriate identity management, secrets management, encryption, network controls, retention/deletion policies, monitoring, backups, incident response, and independent security/privacy review.

## Compliance workflow

DataGuard includes versioned control definitions for:

- Québec privacy requirements
- Canadian privacy requirements
- GDPR
- CCPA

These controls are configurable engineering artifacts, **not legal advice or proof of compliance**. Organizational policy and qualified legal/privacy review remain required.

## Testing & quality

The repository includes automated testing across application and security boundaries:

```text
Unit tests
    ↓
API tests
    ↓
Security / authorization tests
    ↓
Domain workflow tests
    ↓
GitHub Actions quality gate
```

The project also includes synthetic demonstration-data generation so that privacy-sensitive workflows can be tested without relying on real personal information.

## Infrastructure

```text
Docker
   │
   ▼
FastAPI application
   │
   ├── PostgreSQL
   ├── Redis
   ├── SQLAlchemy / Alembic
   └── Enterprise Web UI

GitHub Actions
   │
   └── Automated quality checks
```

The repository is structured so that production infrastructure can later be deployed using appropriate managed services and security controls.

## Production readiness boundary

DataGuard is an **enterprise product foundation and evaluation candidate**. It should not be represented as:

- a government-certified system
- an accredited privacy/compliance platform
- proof of legal compliance
- evidence of a government contract
- evidence of customer deployment
- a replacement for qualified privacy/legal review
- an AI/ML system with measured model performance when no such evidence exists

Before processing real sensitive information, the target deployment requires environment-specific security assessment, privacy/legal review, operational validation, appropriate PII evaluation, and acceptance testing.

## Roadmap

Future engineering work can include:

- Reproducible model training/evaluation pipeline
- Versioned PII benchmark dataset
- Precision/recall/F1 and per-class evaluation
- Human-in-the-loop review tooling
- OCR implementation with measured extraction quality
- Asynchronous document processing
- Production observability and alerting
- Managed deployment infrastructure
- Security and privacy validation in the target environment

## License / usage

Review the repository's licensing and dependency terms before commercial use. Sensitive data, credentials, and secrets must never be committed to source control.
