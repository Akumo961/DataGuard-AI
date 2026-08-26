# Phase 13 — Testing

## Repository evidence

The repository contains unit and domain-focused tests covering API behavior, authentication/security, tenancy/database behavior, PII detection, processing, risk, compliance, PIA and audit services, plus architecture contracts.

A GitHub Actions quality workflow is present under `.github/workflows/quality.yml`.

## Limitation

A repository test suite is not equivalent to an independent security assessment, production load test, disaster-recovery exercise or government user-acceptance test. Those produce target-environment evidence and remain separate release gates.
