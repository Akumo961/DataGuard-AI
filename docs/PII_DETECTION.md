# PII detection scope and limitations

DataGuard combines deterministic regex/contextual rules with an optional externally supplied NER adapter. Detection is **best-effort discovery**, not a completeness guarantee.

## PERSON

The baseline detector recognizes high-signal forms such as:

- labeled names (`Nom`, `Prénom`, `Name`);
- honorific + name (`Mme Marie Tremblay`, `Dr Jean Tremblay`);
- explicit name-context phrases (`first name`, `last name`, `nom de famille`).

It intentionally does **not** classify every sequence of capitalized words as a person. This reduces false positives for organizations, departments and document titles.

## ADDRESS

The detector recognizes high-signal civic-address patterns including street types in English/French and Canadian postal-code context. It does not guarantee detection of every free-form, rural, international, PO-box or malformed address.

## Operational limitation

False negatives and false positives remain possible, especially with OCR errors, unusual formatting, multilingual text and names that resemble ordinary words. Production deployments should evaluate the detector against representative, de-identified customer corpora and monitor precision/recall by PII type.

The optional `NERDetector` is an adapter boundary only; no pretrained NER model is bundled or represented as production-ready. A model must be independently evaluated, mapped to the DataGuard taxonomy and versioned before use.
