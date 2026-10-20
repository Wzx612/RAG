"""Static checks for the no-build acceptance dashboard assets.

These tests parse the files; they do not prove that a browser rendered them.
Browser and container checks are recorded separately in the documentation.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from src.config import PROJECT_ROOT

STATIC_DIR = PROJECT_ROOT / "web" / "acceptance"
INDEX = STATIC_DIR / "index.html"
SCRIPT = STATIC_DIR / "app.js"
STYLES = STATIC_DIR / "styles.css"

REQUIRED_IDS = {
    "overall-status",
    "error-banner",
    "error-text",
    "retry-button",
    "tab-overview",
    "tab-retrieval",
    "tab-index",
    "tab-chunking",
    "tab-rerank",
    "tab-judge",
    "tab-evidence",
    "panel-overview",
    "panel-retrieval",
    "panel-index",
    "panel-chunking",
    "panel-rerank",
    "panel-judge",
    "panel-evidence",
    "overview-meta",
    "summary-cards",
    "checks",
    "artifact-table",
    "dataset",
    "retrieval-content",
    "index-content",
    "chunking-content",
    "rerank-content",
    "judge-content",
    "evidence-content",
}


class _PageParser(HTMLParser):
    """Collect tags, attributes and inline script/style content."""

    def __init__(self) -> None:
        """Initialize empty collections."""
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str]]] = []
        self.ids: set[str] = set()
        self.script_data = ""
        self.style_data = ""
        self.text: list[str] = []
        self._in_script = False
        self._in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record a start tag and its attributes."""
        attributes = {name: value or "" for name, value in attrs}
        self.tags.append((tag, attributes))
        if "id" in attributes:
            self.ids.add(attributes["id"])
        self._in_script = tag == "script"
        self._in_style = tag == "style"

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Record a self-closing tag."""
        attributes = {name: value or "" for name, value in attrs}
        self.tags.append((tag, attributes))
        if "id" in attributes:
            self.ids.add(attributes["id"])
        self._in_script = False
        self._in_style = False

    def handle_endtag(self, tag: str) -> None:
        """Leave a script or style block."""
        if tag == "script":
            self._in_script = False
        if tag == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        """Collect inline script/style data and visible text separately."""
        if self._in_script:
            self.script_data += data
        elif self._in_style:
            self.style_data += data
        else:
            self.text.append(data)


def _parse(path: Path = INDEX) -> _PageParser:
    """Parse one HTML file."""
    parser = _PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def test_document_metadata_and_language() -> None:
    """The page declares its language, viewport and a same-origin CSP."""
    page = _parse()
    html = [attrs for tag, attrs in page.tags if tag == "html"]
    assert html and html[0]["lang"] == "zh-CN"
    meta = {
        attrs.get("name", attrs.get("http-equiv", "")): attrs.get("content", "")
        for tag, attrs in page.tags
        if tag == "meta"
    }
    assert "width=device-width" in meta["viewport"]
    policy = meta["Content-Security-Policy"]
    assert "default-src 'self'" in policy
    assert "script-src 'self'" in policy
    assert "style-src 'self'" in policy
    assert "base-uri 'none'" in policy
    assert "form-action 'none'" in policy


def test_required_ids_exist() -> None:
    """Every element the script addresses is present in the shell."""
    page = _parse()
    assert page.ids >= REQUIRED_IDS


def test_assets_are_local_and_not_inline() -> None:
    """Only same-origin files are loaded; no inline script or event handler."""
    page = _parse()
    scripts = [attrs for tag, attrs in page.tags if tag == "script"]
    stylesheets = [attrs for tag, attrs in page.tags if tag == "link"]
    assert scripts and all(attrs.get("src") == "app.js" for attrs in scripts)
    assert stylesheets and all(attrs.get("href") == "styles.css" for attrs in stylesheets)
    event_attributes = [
        name
        for _, attrs in page.tags
        for name in attrs
        if name.lower().startswith("on")
    ]
    assert event_attributes == []
    assert page.script_data.strip() == ""
    assert page.style_data.strip() == ""


def test_no_external_urls_in_any_asset() -> None:
    """No CDN, font, image or stylesheet is fetched from the network."""
    for path in (INDEX, SCRIPT, STYLES):
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text
        assert "https://" not in text
        assert "//cdn" not in text
    css = STYLES.read_text(encoding="utf-8")
    assert "@import" not in css
    assert "url(" not in css


def test_javascript_is_text_only_and_points_at_both_endpoints() -> None:
    """The script writes through textContent and calls both read-only APIs."""
    script = SCRIPT.read_text(encoding="utf-8")
    assert '"textContent"' in script or "textContent" in script
    for forbidden in ("innerHTML", "insertAdjacentHTML", "document.write", "outerHTML"):
        assert forbidden not in script
    assert re.search(r"\beval\s*\(", script) is None
    assert "/acceptance/api/summary" in script
    assert "/acceptance/api/judge/queries" in script
    assert "fetch(" in script
    assert "SECRET" not in script


def test_page_states_read_only_and_not_a_human_verdict() -> None:
    """The visible text does not present the page as a sign-off."""
    page = _parse()
    text = "".join(page.text)
    assert "只读" in text
    assert "不是人工验收结论" in text
    assert "NOT RUN" in text
    assert "重试" in text


def test_script_has_explicit_status_vocabulary() -> None:
    """Statuses are text labels, not colour alone."""
    script = SCRIPT.read_text(encoding="utf-8")
    for status in (
        "PASS",
        "PARTIAL",
        "FAIL",
        "BLOCKED",
        "MISSING",
        "NOT RUN",
        "NOT APPLICABLE",
    ):
        assert status in script
