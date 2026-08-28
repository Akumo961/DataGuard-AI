# DataGuard Québec

> **Enterprise AI-assisted privacy, PII discovery, risk assessment, and compliance workflow platform.**

DataGuard Québec is a production-oriented engineering foundation for discovering and classifying sensitive information, assessing privacy risk, supporting Privacy Impact Assessment (PIA) workflows, applying configurable compliance controls, and maintaining auditable evidence.

The project is designed around a clear principle: **privacy automation should be explainable, testable, tenant-aware, and reviewable by humans.**

> **Important:** DataGuard does not automatically establish legal compliance, certification, government approval, or regulatory conformity. Compliance controls and reports require organizational and qualified legal/privacy review.

## Why DataGuard?

Privacy teams often need to identify sensitive information across documents, understand the associated risk, map findings to applicable controls, and preserve evidence for review. DataGuard demonstrates how these workflows can be implemented as an integrated AI-enabled software platform rather than as a standalone model.

## Engineering highlights

- FastAPI application with versioned APIs
- Configurable multi-layer PII detection architecture
- Deterministic and explainable privacy risk scoring
- PostgreSQL / SQLAlchemy / Alembic persistence
- PostgreSQL Row-Level Security (RLS) migration support for tenant isolation
- JWT authentication and tenant-aware authorization boundaries
- PIA workflow and state-transition model
- Versioned Québec, Canadian, GDPR, and CCPA control definitions
- Document extraction, validation, and OCR interfaces
- Audit and evidence service foundation
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
          Authentication    Application     Audit / Evidence
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

## AI / ML architecture

DataGuard is intentionally designed so that deterministic controls and AI/ML components can coexist without making model output the sole source of truth.

The current repository provides the detection architecture, privacy taxonomy, workflow boundaries, and evaluation foundations required for future production ML adoption. Model-backed detection should be introduced only with reproducible datasets, model provenance, measured precision/recall/F1, per-class error analysis, and human review.

### No unsupported accuracy claims

The repository does **not** claim a production PII detection accuracy such as "95%+" because the current repository does not contain sufficient reproducible evidence to support that figure. Historical prototype artifacts containing empty model/data files are not treated as trained production models or validation evidence.

This is intentional: **measured AI performance is more valuable than an unverified benchmark claim.**

## Security & privacy

Security is treated as a system property rather than a prompt or model feature.

Current engineering controls include:

- JWT authentication boundary
- Tenant-aware authorization
- PostgreSQL RLS migration support
- Input validation through Pydantic contracts
- Audit/evidence foundations
- Environment-based configuration
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
   ├── SQLAlchemy / Alembic
   └── Enterprise Web UI

GitHub Actions
   │
   └── Automated quality checks
```

The repository is structured so that production infrastructure can later be deployed using appropriate managed services and security controls.

## Repository structure

```text
DataGuard/
├── dataguard/                 # Core application
├── docs/                      # Architecture, security, privacy, deployment,
│                              # procurement, testing, governance, etc.
├── tests/                     # Unit/API/security/domain tests
├── migrations/                # Database migrations
├── .github/                   # CI workflows
├── docker-compose.local.yml   # Local development environment
└── README.md
```

## Documentation

The repository contains dedicated documentation covering:

- Architecture
- Security architecture
- Threat model
- Privacy architecture
- Compliance controls
- Privacy Impact Assessment (PIA) workflows
- Deployment
- Testing
- Procurement considerations
- Release governance
- Government demonstration guidance

Start with the documents under `docs/` for the detailed engineering and operational view.

## Local development

See `docs/DEPLOYMENT.md` and `docker-compose.local.yml` for the supported local setup.

Use **synthetic demonstration data only** during development. Do not upload real personal information to an unapproved environment.

## Production readiness boundary

DataGuard is an **enterprise product foundation and evaluation candidate**. It should not be represented as:

- a government-certified system
- an accredited privacy/compliance platform
- proof of legal compliance
- evidence of a government contract
- evidence of customer deployment
- a replacement for qualified privacy/legal review

Before processing real sensitive information, the target deployment requires environment-specific security assessment, privacy/legal review, operational validation, appropriate ML evaluation, and acceptance testing.

## Engineering focus

DataGuard demonstrates practical AI engineering across:

**AI/ML architecture · privacy engineering · PII detection · risk scoring · compliance workflows · secure APIs · multi-tenancy · PostgreSQL · testing · Docker · CI/CD · auditability**

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
