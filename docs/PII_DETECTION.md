# PII Detection Engine

DataGuard uses a layered detection architecture. The current production-safe baseline is deterministic regex detection plus validation. An optional NER adapter and ensemble boundary are provided for independently evaluated models.

## Layers

1. Pattern detection: email, phone, IP, credit-card-shaped values, SIN-shaped values, passport-shaped values and ISO-like dates.
2. NER adapter: integrates an externally evaluated NER model through an explicit taxonomy mapping.
3. Transformer/classification adapters: reserved behind the same detector contract; no unsupported production model is bundled.
4. Contextual validation: removes obvious invalid candidates.
5. Confidence scoring: every detection carries a bounded confidence score and detector provenance.
6. Rule validation: credit-card candidates use Luhn validation; other identifiers use conservative structural checks.
7. Human review: the domain model remains compatible with review workflows; review persistence belongs to later workflow phases.

## Privacy

The detector operates on text and returns offsets, type, confidence and provenance. Callers should avoid persisting raw values unless required by an approved purpose. Production integrations should prefer redaction or tokenization before persistence.

## Limitations

Regex cannot establish identity or legal sensitivity by itself. Passport, government-ID and date patterns are intentionally conservative heuristics and require contextual/model confirmation. No accuracy or recall claim is made by this implementation. Evaluation datasets and reproducible benchmark results are part of the ML validation workstream.
