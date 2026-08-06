import hashlib
import re
from html.parser import HTMLParser
from typing import Any, ClassVar


class _TextExtractor(HTMLParser):
    """Small dependency-free HTML-to-text parser for news payloads."""

    _BLOCK_TAGS: ClassVar[frozenset[str]] = frozenset({
        "article", "blockquote", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "p", "section", "table", "td", "th", "tr",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def clean_html_text(raw_html: Any) -> str:
    if raw_html is None:
        return ""
    parser = _TextExtractor()
    parser.feed(str(raw_html))
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def compact_text(value: Any, limit: int | None = None) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if limit and len(text) > limit:
        return text[:limit].rstrip()
    return text


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def stable_id(*parts: object, prefix: str = "") -> str:
    payload = "\n".join(str(part or "") for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def make_prompt(instruction: str, input_text: str) -> str:
    return f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "
