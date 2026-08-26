# Phase 11 — Enterprise Frontend

## Delivered

DataGuard now includes a responsive enterprise dashboard under `dataguard/frontend/`, served by the FastAPI application at `/` and `/frontend/*`.

### Views

- Executive/privacy overview
- Data discovery
- PII findings
- PIA workflow status
- Remediation
- Audit & evidence

### Analysis experience

The dashboard calls the real `/api/v1/analyze` endpoint. It presents risk score, level, contributing factors, recommendations and redacted PII findings. Sensitive raw values are not intentionally rendered by the UI.

A short-lived in-memory Bearer token field is provided because Phase 10 exposes an authentication boundary rather than a complete enterprise identity provider. The UI does not persist the token in localStorage or cookies.

## Design principles

- responsive layout for desktop and tablet
- accessible semantic controls and status messaging
- no fabricated counts or audit events
- explicit indication when a capability is not yet connected
- no third-party CDN dependency
- no production secrets embedded in frontend assets

## Deliberate boundaries

This phase establishes the enterprise frontend architecture and a functional analysis workflow. It does not claim that authentication, PIA persistence, remediation persistence, audit/evidence storage, enterprise SSO, or all connectors are complete. Those require the corresponding backend capabilities and later integration work.

## Local use

Start the FastAPI application with the project's normal Uvicorn command, then open `/`. Configure a valid development Bearer token for the authenticated analysis endpoint. Production deployments must use the approved OIDC/JWKS configuration documented by the security architecture.
