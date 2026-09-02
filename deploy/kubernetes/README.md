# DataGuard Kubernetes / private-cloud runbook

## Preconditions

- A private Kubernetes cluster is available.
- `dataguard-api-secrets` is provisioned by an external secret manager; secrets are never committed here.
- The container image has been scanned and promoted by the deployment system.

## Upgrade

1. Apply the new image digest, not a mutable tag.
2. Run `kubectl apply -f deploy/kubernetes/`.
3. Wait for `kubectl rollout status deployment/dataguard-api -n dataguard`.
4. Verify readiness and liveness probes and application health.
5. Confirm database migrations completed before serving traffic.

## Rollback

1. Stop promotion if readiness or error-rate alerts fire.
2. Run `kubectl rollout history deployment/dataguard-api -n dataguard`.
3. Roll back with `kubectl rollout undo deployment/dataguard-api -n dataguard --to-revision=<known-good-revision>`.
4. Wait for rollout completion and verify `/health/ready`.
5. Record the incident and migration compatibility decision before retrying.

The manifest uses a RollingUpdate strategy with revision history so Kubernetes can perform the mechanical rollback. A real private-cloud upgrade/rollback exercise is still required before claiming operational evidence.
