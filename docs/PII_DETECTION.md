# PII detection scope and limitations

DataGuard combines deterministic regex/contextual rules with a multilingual NER detector. Detection is **best-effort discovery**, not a completeness guarantee.

## PERSON

The baseline detector recognizes high-signal forms such as:

- labeled names (`Nom`, `Prénom`, `Name`);
- honorific + name (`Mme Marie Tremblay`, `Dr Jean Tremblay`);
- explicit name-context phrases (`first name`, `last name`, `nom de famille`).

For deployments with `DATAGUARD_NER_MODEL=xx_ent_wiki_sm`, DataGuard also loads the lightweight spaCy `xx_ent_wiki_sm` WikiNER model and maps its `PER` label to `PERSON`. This adds plain narrative name detection without replacing the deterministic rules. The model is multilingual, but its output is not a government-use accuracy guarantee.

## ADDRESS

The deterministic detector recognizes high-signal civic-address patterns including street types in English/French and Canadian postal-code context. The NER model is **not** treated as an address detector. DataGuard does not guarantee detection of every free-form, rural, international, PO-box or malformed address.

## NER evaluation

The checked-in synthetic benchmark is `dataguard/evaluation/ner_benchmark.py`. It uses fictional Québec/English names, locations and address-containing prose and computes exact-span precision, recall and F1 per entity type at runtime. No fixed accuracy number is claimed without running that script against the pinned model.

With the optional NER dependency installed and the model available:

```bash
python -m pip install -e '.[ner]'
python -m spacy download xx_ent_wiki_sm
python -m dataguard.evaluation.ner_benchmark --model xx_ent_wiki_sm
```

The Docker image pins the model wheel by SHA-256 before installation. Production deployments should still evaluate the detector against representative, de-identified customer corpora and monitor precision/recall by PII type.

## Operational limitation

False negatives and false positives remain possible, especially with OCR errors, unusual formatting, multilingual text, names that resemble ordinary words, and Québec-specific naming conventions. Human review remains necessary for consequential privacy decisions.
