from html.parser import HTMLParser
from pathlib import Path


class _HTMLAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.labels_for: set[str] = set()
        self.form_controls: list[tuple[str, str | None]] = []
        self.main_count = 0
        self.lang: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        data = dict(attrs)
        if tag == "html":
            self.lang = data.get("lang")
        if "id" in data:
            self.ids.add(data["id"])
        if tag == "label" and data.get("for"):
            self.labels_for.add(data["for"])
        if tag in {"input", "textarea", "select"}:
            self.form_controls.append((tag, data.get("id")))
        if tag == "main":
            self.main_count += 1


def test_frontend_has_basic_wcag_semantics() -> None:
    parser = _HTMLAudit()
    parser.feed((Path(__file__).parents[2] / "frontend" / "index.html").read_text(encoding="utf-8"))
    assert parser.lang == "fr-CA"
    assert parser.main_count == 1
    assert "main-content" in parser.ids
    assert {control_id for _, control_id in parser.form_controls if control_id} <= parser.ids
    assert {"role", "input", "access", "exposure", "encrypted", "token"} <= parser.labels_for
