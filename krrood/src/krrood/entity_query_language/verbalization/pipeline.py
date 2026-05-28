"""
Verbalization pipeline — combines verbalizer + renderer for formatted output.

:class:`VerbalizationPipeline` is the high-level entry point for coloured,
hierarchical, or hyperlinked output.  Factory class methods cover the most
common configurations:

* :meth:`~VerbalizationPipeline.plain` — prose, no colour.
* :meth:`~VerbalizationPipeline.ansi` — ANSI true-colour terminal output.
* :meth:`~VerbalizationPipeline.html` — HTML ``<span>`` colours for Jupyter.

All accept an optional *link_resolver* for source hyperlinks.
"""

from __future__ import annotations

import logging
import tempfile
import webbrowser
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.rendering.formatter import (
    ANSIFormatter,
    HTMLFormatter,
    PlainFormatter,
    _detect_osc8_support,
)
from krrood.entity_query_language.verbalization.rendering.renderer import (
    FragmentRenderer,
    HierarchicalRenderer,
    ParagraphRenderer,
)
from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
from krrood.entity_query_language.query.query import Query

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.rendering.source_link_resolver import SourceLinkResolver

_log = logging.getLogger(__name__)

_HTML_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: monospace;
    font-size: 14px;
    padding: 1.5em;
    line-height: 1.8;
  }}
  a {{ color: inherit; }}
</style>
</head>
<body>{body}</body>
</html>
"""

# Inline dark wrapper for HTML cell output (Jupyter / built docs). Mirrors
# _HTML_PAGE_TEMPLATE so the colors read correctly in both environments.
_HTML_CELL_WRAPPER = (
    '<div style="background:#1e1e1e;color:#d4d4d4;font-family:monospace;'
    'font-size:14px;padding:0.75em;border-radius:0.4em;line-height:1.8;">'
    "{body}"
    "</div>"
)


def _is_ipython() -> bool:
    """Return ``True`` when running inside an IPython / Jupyter session."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


@dataclass
class VerbalizationPipeline:
    """
    Combines an :class:`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer`
    (fragment builder) with a
    :class:`~krrood.entity_query_language.verbalization.rendering.renderer.FragmentRenderer`
    (format + colour) to produce a final string.

    Usage::

        pipeline = VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter()))
        text = pipeline.verbalize(query)

    Factory class methods cover the most common configurations:

    * :meth:`plain` — no colour, paragraph prose (default for
      :func:`~krrood.entity_query_language.verbalization.verbalizer.verbalize_expression`).
    * :meth:`ansi`  — ANSI true-colour terminal output.
    * :meth:`html`  — HTML ``<span>`` colours for Jupyter / inline HTML.

    All factory methods accept an optional *link_resolver* that maps class and
    attribute names to hyperlinks.  Built-in resolver:

    * :class:`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.AutoAPIResolver`
      — Sphinx AutoAPI documentation pages (local build or hosted).
    """

    renderer: FragmentRenderer = field(default_factory=ParagraphRenderer)
    """Renderer used to convert the fragment tree to a string."""

    _verbalizer: EQLVerbalizer = field(default_factory=EQLVerbalizer, init=False)
    """The verbalizer that builds fragment trees from EQL expressions."""

    def verbalize(self, expression) -> str:
        """
        Verbalize *expression* to a string using this pipeline's renderer.

        :param expression: Any EQL expression or :class:`~krrood.entity_query_language.query.query.Query`.
        :returns: Formatted natural-language string (plain, ANSI, or HTML depending on renderer).
        :rtype: str
        """
        if isinstance(expression, Query):
            expression.build()
        fragment = self._verbalizer.build(expression)
        return self.verbalize_fragment(fragment)

    def _is_html_renderer(self) -> bool:
        """Return ``True`` when this pipeline's renderer uses :class:`HTMLFormatter`."""
        return isinstance(getattr(self.renderer, "_formatter", None), HTMLFormatter)

    def verbalize_fragment(self, fragment: VerbFragment) -> str:
        """
        Render a pre-built :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
        using this pipeline's renderer.

        HTML pipelines wrap the result in a dark ``<div>`` suitable for Jupyter output.

        :param fragment: Root of the fragment tree to render.
        :type fragment: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        :returns: Formatted string.
        :rtype: str
        """
        result = self.renderer.render(fragment)
        if self._is_html_renderer():
            return _HTML_CELL_WRAPPER.format(body=result)
        return result

    def display(self, expression) -> None:
        """
        Render *expression* and display it in the current environment.

        * **Jupyter / IPython** — renders inline via ``IPython.display.HTML``.
        * **Elsewhere** — writes a temporary ``.html`` file and opens it in the
          default system browser.

        Designed for use with :meth:`html` pipelines.  Calling it on a plain-text
        or ANSI pipeline will open a browser tab with raw text.

        :param expression: Any EQL expression or :class:`~krrood.entity_query_language.query.query.Query`.
        """
        self.display_fragment(self._verbalizer.build(expression))

    def display_fragment(self, fragment: VerbFragment) -> None:
        """
        Display a pre-built :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
        — same environment routing as :meth:`display`.

        :param fragment: Root of the fragment tree to display.
        :type fragment: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        raw_html = self.renderer.render(fragment)
        if _is_ipython():
            from IPython.display import display as _ipython_display, HTML
            wrapped = _HTML_CELL_WRAPPER.format(body=raw_html) if self._is_html_renderer() else raw_html
            _ipython_display(HTML(wrapped))
            return
        full_page = _HTML_PAGE_TEMPLATE.format(body=raw_html)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(full_page)
            tmp_path = f.name
        webbrowser.open(f"file://{tmp_path}")

    # ── Factories ──────────────────────────────────────────────────────────────

    @classmethod
    def plain(cls) -> VerbalizationPipeline:
        """
        Create a plain-text, paragraph-prose pipeline with no colour markup.

        :returns: A :class:`VerbalizationPipeline` backed by
            :class:`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer`
            and :class:`~krrood.entity_query_language.verbalization.rendering.formatter.PlainFormatter`.
        :rtype: VerbalizationPipeline
        """
        return cls(ParagraphRenderer(PlainFormatter()))

    @classmethod
    def ansi(
        cls,
        hierarchical: bool = False,
        link_resolver: Optional["SourceLinkResolver"] = None,
    ) -> VerbalizationPipeline:
        """
        Create an ANSI true-colour (24-bit) pipeline for terminal display.

        When *link_resolver* is provided and the terminal supports OSC 8
        hyperlinks (detected via environment variables), class and attribute names
        become clickable.  On unsupported terminals a warning is logged and the
        resolver is silently disabled.

        :param hierarchical: When ``True`` use
            :class:`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer`
            (indented bullets); otherwise use paragraph prose.
        :type hierarchical: bool
        :param link_resolver: Optional resolver mapping source references to URLs.
        :type link_resolver: ~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver or None
        :returns: An ANSI-coloured :class:`VerbalizationPipeline`.
        :rtype: VerbalizationPipeline
        """
        formatter = ANSIFormatter()
        if link_resolver is not None and not _detect_osc8_support():
            _log.warning(
                "The current terminal does not appear to support OSC 8 hyperlinks "
                "(VTE_VERSION / TERM_PROGRAM / TERM not recognised). "
                "link_resolver will be ignored for ANSI output."
            )
            link_resolver = None
        renderer: FragmentRenderer = (
            HierarchicalRenderer(formatter, link_resolver)
            if hierarchical
            else ParagraphRenderer(formatter, link_resolver)
        )
        return cls(renderer)

    @classmethod
    def html(
        cls,
        hierarchical: bool = False,
        link_resolver: Optional["SourceLinkResolver"] = None,
    ) -> VerbalizationPipeline:
        """
        Create an HTML ``<span>`` colour pipeline for Jupyter / inline-HTML rendering.

        When *link_resolver* is provided, class and attribute names are wrapped in
        ``<a href="…">`` anchors pointing to documentation pages.

        :param hierarchical: When ``True`` use
            :class:`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer`
            (indented bullets); otherwise use paragraph prose.
        :type hierarchical: bool
        :param link_resolver: Optional resolver mapping source references to URLs.
        :type link_resolver: ~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver or None
        :returns: An HTML-coloured :class:`VerbalizationPipeline`.
        :rtype: VerbalizationPipeline
        """
        formatter = HTMLFormatter()
        renderer = (
            HierarchicalRenderer(formatter, link_resolver)
            if hierarchical
            else ParagraphRenderer(formatter, link_resolver)
        )
        return cls(renderer)
