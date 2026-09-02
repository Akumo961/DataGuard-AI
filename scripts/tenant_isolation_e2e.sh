#!/usr/bin/env bash
set -euo pipefail

base="${DATAGUARD_BASE_URL:-http://127.0.0.1:8000}"
secret="${DATAGUARD_JWT_SECRET:-ci-jwt-secret-with-more-than-32-characters}"

jwt() {
  local org="$1" sub="$2" role="$3"
  docker compose exec -T \
    -e "ORG=$org" -e "SUB=$sub" -e "ROLE=$role" -e "SECRET=$secret" \
    api python -c 'import jwt; from datetime import datetime,timedelta,timezone; import os; n=datetime.now(timezone.utc); print(jwt.encode({"sub":os.environ["SUB"],"org":os.environ["ORG"],"roles":[os.environ["ROLE"]],"iat":n,"exp":n+timedelta(minutes=10),"jti":os.environ["SUB"]+"-e2e"}, os.environ["SECRET"], algorithm="HS256"))'
}

A="$(jwt 00000000-0000-0000-0000-000000000042 tenant-a analyst)"
B="$(jwt 00000000-0000-0000-0000-000000000043 tenant-b analyst)"
AUTH_A=(-H "Authorization: Bearer $A")
AUTH_B=(-H "Authorization: Bearer $B")
JSON=(-H 'Content-Type: application/json')

analysis="$(curl --fail --silent "${AUTH_A[@]}" "${JSON[@]}" -X POST "$base/api/v1/analyze" -d '{"text":"alice@example.com","purpose_defined":true,"encrypted_at_rest":true,"access_scope":"internal","data_location":"quebec"}')"
analysis_id="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["analysis_id"])' <<<"$analysis")"

test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_B[@]}" "$base/api/v1/analyses/$analysis_id")" = 404
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_B[@]}" -X DELETE "$base/api/v1/analyses/$analysis_id")" = 404
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_A[@]}" "$base/api/v1/analyses/$analysis_id")" = 200

findings="$(curl --fail --silent "${AUTH_A[@]}" "$base/api/v1/findings?analysis_id=$analysis_id")"
finding_id="$(python -c 'import json,sys; x=json.loads(sys.stdin.read()); print(x[0]["id"])' <<<"$findings")"
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_B[@]}" "$base/api/v1/findings/$finding_id")" = 404
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_B[@]}" -X PATCH "$base/api/v1/findings/$finding_id?status=RESOLVED")" = 404
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_A[@]}" -X PATCH "$base/api/v1/findings/$finding_id?status=RESOLVED")" = 200
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_A[@]}" -X DELETE "$base/api/v1/findings/$finding_id")" = 204

pia="$(curl --fail --silent "${AUTH_A[@]}" "${JSON[@]}" -X POST "$base/api/v1/pias" -d '{"project_name":"Tenant A PIA","data_subjects":["employees"],"purposes":["testing"],"vendors":["synthetic"],"jurisdictions":["CA-QC"],"safeguards":["encryption"]}')"
pia_id="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["id"])' <<<"$pia")"
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_B[@]}" "$base/api/v1/pias/$pia_id")" = 404
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_B[@]}" -X DELETE "$base/api/v1/pias/$pia_id")" = 404
curl --fail --silent "${AUTH_A[@]}" "${JSON[@]}" -X POST "$base/api/v1/pias/$pia_id/transition" -d '{"target":"IN_REVIEW","reason":"Review started"}' >/dev/null
curl --fail --silent "${AUTH_A[@]}" "${JSON[@]}" -X POST "$base/api/v1/pias/$pia_id/transition" -d '{"target":"APPROVED","reason":"Approved after documented privacy review"}' >/dev/null

action="$(curl --fail --silent "${AUTH_A[@]}" "${JSON[@]}" -X POST "$base/api/v1/remediations" -d '{"title":"Tenant A remediation","description":"Synthetic remediation","priority":"HIGH","sla_hours":24}')"
remediation_id="$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["id"])' <<<"$action")"
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_B[@]}" -X PATCH "$base/api/v1/remediations/$remediation_id" "${JSON[@]}" -d '{"status":"IN_PROGRESS"}')" = 404
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_A[@]}" -X PATCH "$base/api/v1/remediations/$remediation_id" "${JSON[@]}" -d '{"status":"IN_PROGRESS","evidence":{"ticket":"synthetic-123"}}')" = 200
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_B[@]}" -X DELETE "$base/api/v1/remediations/$remediation_id")" = 404
test "$(curl --silent -o /dev/null -w '%{http_code}' "${AUTH_A[@]}" -X DELETE "$base/api/v1/remediations/$remediation_id")" = 204

printf 'Tenant isolation matrix passed for read/create/update/delete/inference paths.\n'
