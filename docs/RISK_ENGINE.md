# DataGuard Risk Engine

The Phase 6 risk engine is deterministic and explainable. It is an advisory privacy/security risk signal, not a legal conclusion, compliance certification, or automated decision about an individual's rights.

## Inputs

- detected PII classes and detector confidence
- quantity of detections
- data location
- access scope
- retention period
- encryption at rest confirmation
- documented purpose confirmation
- exposure
- optional framework identifier
- organization policy multiplier

## Output

Every assessment returns a 0–100 score, LOW/MEDIUM/HIGH/CRITICAL level, contributing factors with points and detail, an explanation, and remediation recommendations.

## Method

The baseline score starts from the highest configured sensitivity class, adds bounded quantity and confidence contributions, then adds explicit control/exposure factors. The final value is clamped to 0–100 and mapped to levels: <30 LOW, 30–59.99 MEDIUM, 60–79.99 HIGH, >=80 CRITICAL.

The formula is intentionally inspectable and versionable. It should be calibrated against organizational risk appetite and historical findings before being used as a production decision-support control.

## Limitations

The engine does not infer legal applicability. A framework field is contextual metadata only until the compliance rules engine supplies verified, versioned controls. Missing context increases uncertainty and can conservatively increase the score. Scores must be reviewed by authorized privacy/security personnel for consequential decisions.
