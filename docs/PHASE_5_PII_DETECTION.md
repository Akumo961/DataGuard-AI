# Phase 5 — PII Detection Engine

## Delivered

- Composable detector contract.
- Deterministic regex detector for common PII patterns.
- Luhn validation for credit-card-shaped candidates.
- Conservative structural validation for SIN-shaped candidates.
- Context validation layer.
- Ensemble confidence aggregation.
- Optional NER adapter contract with explicit taxonomy mapping.
- Detection pipeline with detector provenance and confidence.
- Unit tests for common detections, validation, Luhn behavior and ensemble aggregation.

## Deliberate limitations

The repository does not contain a production-trained transformer/NER model or a representative labeled corpus. Therefore DataGuard makes no precision, recall, F1 or accuracy claim. The NER adapter is an integration boundary, not a claim that a model is deployed.

Human review, persistent findings and correction workflows are implemented in later product phases. OCR and document extraction belong to Phase 9.

## Privacy requirement

Raw detected values should not be persisted by default. Production persistence should prefer offsets, hashes/tokens or redacted representations unless an approved purpose requires the original value.

## Taxonomy

The domain taxonomy includes PERSON, EMAIL, PHONE, ADDRESS, DATE_OF_BIRTH, GOVERNMENT_ID, PASSPORT, DRIVER_LICENSE, HEALTH_INFORMATION, FINANCIAL_INFORMATION, BANK_ACCOUNT, CREDIT_CARD, IP_ADDRESS, LOCATION, ORGANIZATION, EMPLOYEE_ID, CUSTOMER_ID, TAX_ID, SOCIAL_INSURANCE_NUMBER, BIOMETRIC_DATA and OTHER_SENSITIVE_INFORMATION.
