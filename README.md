# DataGuard Québec

**AI-assisted privacy, personal-information discovery, risk assessment and compliance platform.**

DataGuard provides an enterprise-oriented foundation for discovering and classifying sensitive information, assessing privacy risk, supporting PIA workflows, applying configurable compliance controls, and maintaining audit evidence.

> **Important:** DataGuard does not automatically establish legal compliance, certification, government approval, or regulatory conformity. Compliance controls and reports require organizational and qualified legal/privacy review.

## Current architecture

```text
FastAPI API
   │
   ├── Authentication / tenant context
   ├── Application services
   ├── PII detection pipeline
   ├── Risk engine
   ├── Compliance rules engine
   ├── PIA workflow
   ├── Audit/evidence services
   └── Document processing
          │
          ├── extraction
          ├── validation
          └── OCR boundary

PostgreSQL / SQLAlchemy / Alembic

Enterprise web UI

Docker / CI quality pipeline
```

## Verified capabilities in this repository

- FastAPI application and versioned API
- Pydantic contracts and validation
- JWT authentication boundary with tenant context
- PostgreSQL/SQLAlchemy persistence foundation
- tenant isolation controls including PostgreSQL RLS migrations
- configurable PII taxonomy and multi-layer detection architecture
- deterministic privacy risk scoring with explainable contributing factors
- versioned Québec, Canadian, GDPR and CCPA control files
- PIA workflow model and state transitions
- document extraction/validation/OCR interfaces
- audit/evidence service foundation
- responsive enterprise web dashboard
- automated unit/API/security/domain tests
- Docker and local development configuration
- GitHub Actions quality workflow
- synthetic demonstration-data generator

## AI/ML claims

The repository does **not** contain sufficient reproducible evidence to support a production accuracy claim such as "95%+ accuracy". Historical prototype artifacts include empty model/data files; these are not treated as trained production models or validation evidence.

Production ML adoption requires an approved evaluation dataset, reproducible training/evaluation pipeline, model provenance, precision/recall/F1, per-class error analysis and human review.

## Run locally

See `docs/DEPLOYMENT.md` and `docker-compose.local.yml` for the supported local setup.

The primary application entry point is the FastAPI application under `dataguard/`. The enterprise UI is served by the application at `/` when configured for local use.

## Documentation

See `docs/` for architecture, security, threat model, privacy architecture, compliance controls, PIA, deployment, testing, procurement, release governance and government demonstration documentation.

## Security and privacy

Do not upload real personal information to an unapproved development environment. Use the synthetic demo data for demonstrations. Production deployments require approved identity, secrets, network, encryption, retention, monitoring, backup and incident-response controls.

## Project status

This repository is an enterprise product foundation and evaluation candidate. It is **not** evidence of a government contract, certification, accreditation, customer deployment or automatic legal compliance. Independent security assessment, target-environment validation, ML validation, legal/privacy review and operational acceptance remain required before a government production deployment.
