"""Small, dependency-free TeX-subset layout for matrix labels.

Qt rich text has no TeX maths layout: nested ``<sup>/<sub>`` tags position a
subscript once relative to the superscript and a second time relative to the
base line.  Matrix expressions only need a deliberately small, predictable
subset (groups plus ``^``/``_``), so using a full browser/LaTeX runtime would
add a large release dependency for very little value.  This module parses that
subset and places each script relative to its *parent atom* with QFontMetrics.

It is intentionally presentation-only.  Symbolic calculation and the LaTeX
source copied from the matrix inspector remain owned by ``model.symbolic``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter
from PySide6.QtWidgets import QGraphicsItem


_MACROS = {
    r"\phi": "φ",
    r"\omega": "ω",
    r"\pi": "π",
    r"\cdot": "·",
    r"\times": "×",
    r"\pm": "±",
    r"\mp": "∓",
    r"\infty": "∞",
    r"\mathrm": "",
    r"\text": "",
    r"\left": "",
    r"\right": "",
}


def normalize_math_source(source: str) -> str:
    """Normalize the TeX-like fragments emitted by HamiVisualizer's model."""
    text = str(source).replace("−", "-")
    for macro, replacement in _MACROS.items():
        text = text.replace(macro, replacement)
    return text


def math_font(pixel_size: int, *, italic: bool = False) -> QFont:
    """Return a stable mathematics face with useful cross-platform fallbacks."""
    font = QFont("Cambria Math")
    if not font.exactMatch():
        font = QFont("STIX Two Math")
    if not font.exactMatch():
        font = QFont("DejaVu Serif")
    if not font.exactMatch():
        font = QFont("Times New Roman")
    font.setPixelSize(max(1, int(pixel_size)))
    font.setStyleHint(QFont.Serif)
    font.setStyleStrategy(QFont.PreferAntialias)
    font.setItalic(bool(italic))
    return font


def _scaled_font(font: QFont, factor: float) -> QFont:
    out = QFont(font)
    pixel = out.pixelSize()
    if pixel > 0:
        out.setPixelSize(max(5, round(pixel * factor)))
    else:
        out.setPointSizeF(max(4.5, out.pointSizeF() * factor))
    return out


@dataclass(frozen=True)
class _Box:
    width: float
    ascent: float
    descent: float


class _Node:
    def box(self, font: QFont) -> _Box:
        raise NotImplementedError

    def draw(self, painter: QPainter, x: float, baseline: float,
             font: QFont, color: QColor) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class _Text(_Node):
    value: str

    def _font(self, font: QFont) -> QFont:
        # TeX mathematics uses italic Latin identifiers by default.  Keep
        # numbers, signs and Greek symbols upright so numeric matrices remain
        # easy to scan and the result still resembles conventional notation.
        return math_font(font.pixelSize(), italic=True) if (
            self.value.isascii() and self.value.isalpha()
        ) else font

    def box(self, font: QFont) -> _Box:
        actual = self._font(font)
        metrics = QFontMetricsF(actual)
        return _Box(metrics.horizontalAdvance(self.value), metrics.ascent(), metrics.descent())

    def draw(self, painter: QPainter, x: float, baseline: float,
             font: QFont, color: QColor) -> None:
        painter.setFont(self._font(font))
        painter.setPen(color)
        painter.drawText(x, baseline, self.value)


@dataclass(frozen=True)
class _Group(_Node):
    children: tuple[_Node, ...] = ()

    def box(self, font: QFont) -> _Box:
        if not self.children:
            metrics = QFontMetricsF(font)
            return _Box(0.0, metrics.ascent(), metrics.descent())
        boxes = [child.box(font) for child in self.children]
        return _Box(
            sum(box.width for box in boxes),
            max(box.ascent for box in boxes),
            max(box.descent for box in boxes),
        )

    def draw(self, painter: QPainter, x: float, baseline: float,
             font: QFont, color: QColor) -> None:
        cursor = float(x)
        for child in self.children:
            child.draw(painter, cursor, baseline, font, color)
            cursor += child.box(font).width


@dataclass(frozen=True)
class _Script(_Node):
    base: _Node
    superscript: _Node | None = None
    subscript: _Node | None = None
    # Values follow TeX's optical relationship, not HTML's nested baseline
    # rule.  In particular the x in e^{ik_x} remains above the main baseline.
    SCRIPT_SCALE: float = field(default=0.68, init=False, repr=False)
    SUPER_RAISE: float = field(default=0.54, init=False, repr=False)
    # A nested subscript such as the ``x`` in ``e^{ik_x}`` is still part of
    # the raised exponent.  The old 0.26 drop was visually closer to HTML's
    # nested ``<sub>`` rule and pushed x toward the primary baseline.  0.16
    # keeps the subscript clearly below k while preserving the optical height
    # of the complete exponent on common Qt math fallbacks.
    SUB_DROP: float = field(default=0.16, init=False, repr=False)

    def _script_font(self, font: QFont) -> QFont:
        return _scaled_font(font, self.SCRIPT_SCALE)

    def _offsets(self, font: QFont) -> tuple[float, float]:
        base = self.base.box(font)
        return -base.ascent * self.SUPER_RAISE, base.ascent * self.SUB_DROP

    def box(self, font: QFont) -> _Box:
        base = self.base.box(font)
        script_font = self._script_font(font)
        super_offset, sub_offset = self._offsets(font)
        width = base.width
        ascent, descent = base.ascent, base.descent
        script_width = 0.0
        if self.superscript is not None:
            sup = self.superscript.box(script_font)
            script_width = max(script_width, sup.width)
            ascent = max(ascent, -super_offset + sup.ascent)
            descent = max(descent, super_offset + sup.descent)
        if self.subscript is not None:
            sub = self.subscript.box(script_font)
            script_width = max(script_width, sub.width)
            ascent = max(ascent, -sub_offset + sub.ascent)
            descent = max(descent, sub_offset + sub.descent)
        # Slight horizontal overlap matches TeX's compact script kerning.
        return _Box(width + script_width * 0.94, ascent, descent)

    def draw(self, painter: QPainter, x: float, baseline: float,
             font: QFont, color: QColor) -> None:
        self.base.draw(painter, x, baseline, font, color)
        base_width = self.base.box(font).width
        script_font = self._script_font(font)
        super_offset, sub_offset = self._offsets(font)
        script_x = x + base_width
        if self.superscript is not None:
            self.superscript.draw(
                painter, script_x, baseline + super_offset, script_font, color,
            )
        if self.subscript is not None:
            self.subscript.draw(
                painter, script_x, baseline + sub_offset, script_font, color,
            )


class _Parser:
    def __init__(self, source: str):
        self.source = normalize_math_source(source)
        self.index = 0

    def parse(self) -> tuple[_Group, ...]:
        lines: list[_Group] = []
        while self.index < len(self.source):
            lines.append(self._sequence(stop_at_newline=True))
            if self.index < len(self.source) and self.source[self.index] == "\n":
                self.index += 1
        return tuple(lines or [_Group()])

    def _sequence(self, *, stop_at_newline: bool = False) -> _Group:
        nodes: list[_Node] = []
        while self.index < len(self.source):
            char = self.source[self.index]
            if char == "}" or (stop_at_newline and char == "\n"):
                break
            if char == "{":
                self.index += 1
                nodes.append(self._sequence())
                if self.index < len(self.source) and self.source[self.index] == "}":
                    self.index += 1
                continue
            if char in "^_":
                self.index += 1
                if not nodes:
                    nodes.append(_Text(char))
                    continue
                script = self._argument()
                base = nodes.pop()
                if isinstance(base, _Script):
                    superscript = script if char == "^" else base.superscript
                    subscript = script if char == "_" else base.subscript
                    nodes.append(_Script(base.base, superscript, subscript))
                elif char == "^":
                    nodes.append(_Script(base, superscript=script))
                else:
                    nodes.append(_Script(base, subscript=script))
                continue
            # Treat a backslash unknown to the normalizer as a literal escape
            # marker only when it precedes a character; this keeps malformed
            # user text visible rather than silently deleting information.
            if char == "\\" and self.index + 1 < len(self.source):
                self.index += 1
                nodes.append(_Text(self.source[self.index]))
                self.index += 1
                continue
            nodes.append(_Text(char))
            self.index += 1
        return _Group(tuple(nodes))

    def _argument(self) -> _Node:
        if self.index >= len(self.source):
            return _Group()
        if self.source[self.index] == "{":
            self.index += 1
            group = self._sequence()
            if self.index < len(self.source) and self.source[self.index] == "}":
                self.index += 1
            return group
        node = _Text(self.source[self.index])
        self.index += 1
        return node


@dataclass(frozen=True)
class MathMetrics:
    width: float
    height: float
    line_boxes: tuple[_Box, ...]


class MathLayout:
    """Parsed maths label; reusable for QGraphicsItem and viewport overlays."""

    def __init__(self, source: str):
        self.source = str(source)
        self.lines = _Parser(self.source).parse()

    def metrics(self, font: QFont) -> MathMetrics:
        boxes = tuple(line.box(font) for line in self.lines)
        leading = max(1.0, QFontMetricsF(font).leading())
        height = sum(box.ascent + box.descent for box in boxes)
        height += leading * max(0, len(boxes) - 1)
        return MathMetrics(max((box.width for box in boxes), default=0.0), height, boxes)

    def draw(self, painter: QPainter, x: float, top: float,
             font: QFont, color: QColor) -> None:
        metrics = self.metrics(font)
        leading = max(1.0, QFontMetricsF(font).leading())
        cursor_top = float(top)
        for line, box in zip(self.lines, metrics.line_boxes):
            line.draw(painter, x, cursor_top + box.ascent, font, color)
            cursor_top += box.ascent + box.descent + leading


class MathTextItem(QGraphicsItem):
    """A scalable scene item backed by :class:`MathLayout`."""

    def __init__(self, source: str, font: QFont, color: QColor, parent=None):
        super().__init__(parent)
        self.layout = MathLayout(source)
        self.font = QFont(font)
        self.color = QColor(color)
        self._metrics = self.layout.metrics(self.font)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(0.0, 0.0, self._metrics.width, self._metrics.height)

    def paint(self, painter: QPainter, option, widget=None):  # noqa: D401
        painter.save()
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        self.layout.draw(painter, 0.0, 0.0, self.font, self.color)
        painter.restore()
