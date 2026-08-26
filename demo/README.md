# DataGuard Québec Demo

This directory defines a safe demonstration environment using synthetic, non-real data.

## Generate dataset

Run:

```text
python demo/synthetic/generate_demo.py
```

The generated files are intentionally small and deterministic. They use `.invalid` email addresses and synthetic identifiers.

## Demo flow

Use `docs/GOVERNMENT_DEMO_SCRIPT.md` for the 15-minute executive walkthrough.

## Restrictions

Do not place real citizen, employee, health, financial or government-identifying information in this directory. Do not use the demo environment as evidence of production security, legal compliance, government approval or model performance.
