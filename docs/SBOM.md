# Software Bill of Materials

DataGuard generates a CycloneDX JSON SBOM in `.github/workflows/sbom.yml` on pushes to protected branches, published releases and manual runs.

The SBOM is uploaded as a GitHub Actions artifact and is not committed to the repository, avoiding stale dependency inventories. Release processes should retain the artifact alongside the release evidence and archive it according to the organization's supply-chain policy.
