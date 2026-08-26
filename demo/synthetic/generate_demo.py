from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent

DOCUMENTS = [
    {"filename": "citizen_service_synthetic.txt", "content": "SYNTHETIC DEMO DATA — NOT REAL PERSONAL INFORMATION\nCitizen: Camille Tremblay\nEmail: camille.tremblay@example.invalid\nPhone: +1 514 555 0101\nGovernment ID: DEMO-QC-000001\nPurpose: municipal service request."},
    {"filename": "employee_record_synthetic.txt", "content": "SYNTHETIC DEMO DATA — NOT REAL PERSONAL INFORMATION\nEmployee: Alex Martin\nEmployee ID: DEMO-EMP-0007\nEmail: alex.martin@example.invalid\nHealth information: synthetic placeholder only."},
]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for document in DOCUMENTS:
        (OUTPUT / document["filename"]).write_text(document["content"], encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(
        json.dumps({"synthetic": True, "documents": [d["filename"] for d in DOCUMENTS]}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
