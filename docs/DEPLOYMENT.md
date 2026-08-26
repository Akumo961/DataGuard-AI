# Deployment

## Local

1. Copy `.env.example` to `.env` and replace placeholder secrets.
2. Run `docker compose up --build`.
3. Verify `/health/live` and `/health/ready`.
4. Run database migrations before using persistent production data.

## Production baseline

Use a managed or operator-controlled PostgreSQL deployment, Redis where required, TLS at the ingress/load-balancer, centralized secrets management, centralized logs/metrics, backups, vulnerability scanning and isolated document-processing workers.

The API container runs as a non-root user and includes a health check. Production credentials must never be committed to Git. Restrict ingress to approved networks and configure explicit allowed origins/hosts.

## Data residency

For Québec/Canadian deployments, select infrastructure and managed services whose contractual, technical and operational characteristics satisfy the organization's approved residency, access, sovereignty and privacy requirements. Data residency is a deployment property; this repository does not certify a provider or jurisdiction.

## Government/private infrastructure

The stack can be deployed on organization-controlled infrastructure or a private cloud. Recommended controls include private database networking, encrypted volumes, centralized key management, immutable backup storage, controlled egress, security monitoring, vulnerability management and documented administrator access.

## Production limitations

This repository does not itself provide a government-certified Kubernetes/HA deployment, WORM storage, enterprise SIEM, managed KMS, OIDC tenant administration or disaster-recovery automation. These must be implemented and validated in the target environment rather than represented as completed features.
