from __future__ import annotations

import ast
import importlib
import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef

_log = logging.getLogger(__name__)


def _find_attribute_line(cls: type, attr_name: str) -> Optional[int]:
    """Return the absolute file line of *attr_name* defined on *cls* via ``AnnAssign``.

    Walks the MRO so inherited dataclass fields are found on the defining class.
    Returns ``None`` when the attribute cannot be located.
    """
    for klass in cls.__mro__:
        if klass is object:
            continue
        try:
            source_lines, class_start = inspect.getsourcelines(klass)
        except (OSError, TypeError):
            continue
        source = "".join(source_lines)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        # The first top-level ClassDef in the snippet is the class itself.
        for top_node in tree.body:
            if isinstance(top_node, ast.ClassDef):
                for item in top_node.body:
                    if (
                        isinstance(item, ast.AnnAssign)
                        and isinstance(item.target, ast.Name)
                        and item.target.id == attr_name
                    ):
                        return class_start + item.lineno - 1
                break  # Only the outer class — do not descend into inner classes.
    return None


class SourceLinkResolver(Protocol):
    """Maps a :class:`SourceRef` to a URL string, or ``None`` when unavailable."""

    def resolve(self, ref: SourceRef) -> Optional[str]:
        ...


@dataclass
class FileURLResolver:
    """Resolves source references to ``file://`` URLs pointing at the local source tree."""

    def resolve(self, ref: SourceRef) -> Optional[str]:
        try:
            path = inspect.getfile(ref.cls)
        except (TypeError, OSError):
            return None
        if ref.attribute is None:
            try:
                _, line = inspect.getsourcelines(ref.cls)
                return f"file://{path}#{line}"
            except (OSError, TypeError):
                return f"file://{path}"
        line = _find_attribute_line(ref.cls, ref.attribute)
        return f"file://{path}#{line}" if line is not None else f"file://{path}"


@dataclass
class JetBrainsResolver:
    """Resolves source references to the ``jetbrains://`` URI scheme registered by JetBrains IDEs.

    Clicking a ``jetbrains://`` link (in the browser or via an OSC 8 terminal hyperlink)
    is handled by the OS URI dispatcher, which routes it to the running JetBrains IDE.
    PyCharm registers this scheme during installation (reliably via JetBrains Toolbox;
    also registered by the standalone installer on most systems).

    The generated URL format is::

        jetbrains://python/navigate/reference?project=.&path=/abs/path/to/file.py:42

    This works for both HTML output (browser click) and ANSI OSC 8 terminal hyperlinks
    (terminal Ctrl+click → ``xdg-open``).  No HTTP server or running process is required
    beyond the IDE itself being installed.
    """

    def resolve(self, ref: SourceRef) -> Optional[str]:
        try:
            path = os.path.abspath(inspect.getfile(ref.cls))
        except (TypeError, OSError):
            return None
        if ref.attribute is None:
            try:
                _, line = inspect.getsourcelines(ref.cls)
            except (OSError, TypeError):
                line = 1
        else:
            line = _find_attribute_line(ref.cls, ref.attribute) or 1
        return f"jetbrains://python/navigate/reference?project=.&path={path}:{line}"


@dataclass
class VSCodeResolver:
    """Resolves source references to the ``vscode://`` URI scheme registered by VS Code.

    Clicking a ``vscode://`` link (in the browser or via an OSC 8 terminal hyperlink)
    is handled by the OS URI dispatcher, which routes it to VS Code at the exact source line.

    The generated URL format is::

        vscode://file//abs/path/to/file.py:42

    This works for both HTML output (browser click) and ANSI OSC 8 terminal hyperlinks.
    No process is required beyond VS Code being installed.
    """

    def resolve(self, ref: SourceRef) -> Optional[str]:
        try:
            path = os.path.abspath(inspect.getfile(ref.cls))
        except (TypeError, OSError):
            return None
        if ref.attribute is None:
            try:
                _, line = inspect.getsourcelines(ref.cls)
            except (OSError, TypeError):
                line = 1
        else:
            line = _find_attribute_line(ref.cls, ref.attribute) or 1
        return f"vscode://file/{path}:{line}"


@dataclass
class AutoAPIResolver:
    """Resolves source references to Sphinx AutoAPI documentation pages.

    *base_url* is the root of the generated docs site, e.g.
    ``https://myproject.readthedocs.io/en/latest`` or a local
    ``http://localhost:63342/project/doc/_build/html``.

    Use :meth:`for_package` to auto-detect the base URL for a locally installed
    package whose docs are served via the JetBrains IDE built-in HTTP server.

    When *html_root* is set (automatically populated by :meth:`for_package`),
    :meth:`resolve` checks that the generated AutoAPI page exists on disk and
    logs a ``WARNING`` if it does not — the class may be missing from the docs
    because they have not been built yet or because AutoAPI excluded it.
    """

    base_url: str
    html_root: Optional[Path] = None

    def resolve(self, ref: SourceRef) -> Optional[str]:
        try:
            module = ref.cls.__module__
            qualname = ref.cls.__qualname__
        except AttributeError:
            return None
        module_path = module.replace(".", "/")
        anchor = f"{module}.{qualname}"
        if ref.attribute is not None:
            anchor = f"{anchor}.{ref.attribute}"
        base = self.base_url.rstrip("/")
        url = f"{base}/autoapi/{module_path}/index.html#{anchor}"
        if self.html_root is not None:
            page = self.html_root / "autoapi" / module_path / "index.html"
            if not page.exists():
                _log.warning(
                    "AutoAPI page for %s.%s does not exist at %s — "
                    "the class may be missing from the docs; "
                    "try rebuilding: sphinx-build doc doc/_build/html",
                    module,
                    qualname,
                    page,
                )
        return url

    @classmethod
    def for_package(cls, package_name: str, port: int = 63342) -> "AutoAPIResolver":
        """Build an :class:`AutoAPIResolver` for *package_name*'s locally built Sphinx docs.

        The base URL targets the JetBrains IDE built-in HTTP server using this algorithm:

        1. Import *package_name* to locate its source tree.
        2. Walk up to the directory containing ``pyproject.toml`` (the package root).
        3. Expect the Sphinx HTML output at ``{package_root}/doc/_build/html``.
        4. Walk up to the git root (directory containing ``.git``).
        5. Construct ``http://localhost:{port}/{git_root_name}/{relative_html_path}``.

        :raises ImportError: if *package_name* cannot be imported.
        :raises FileNotFoundError: if ``doc/_build/html`` does not exist —
            build the docs first with ``sphinx-build doc doc/_build/html``.
        """
        try:
            pkg = importlib.import_module(package_name)
        except ImportError as exc:
            raise ImportError(f"Cannot import package {package_name!r}: {exc}") from exc

        pkg_file = Path(inspect.getfile(pkg)).resolve()

        package_root: Optional[Path] = None
        for parent in pkg_file.parents:
            if (parent / "pyproject.toml").exists():
                package_root = parent
                break
        if package_root is None:
            raise FileNotFoundError(f"No pyproject.toml found in any parent of {pkg_file}")

        html_root = package_root / "doc" / "_build" / "html"
        if not html_root.exists():
            raise FileNotFoundError(
                f"Sphinx HTML output not found at {html_root}. "
                f"Build the docs first: sphinx-build doc doc/_build/html"
            )

        git_root: Optional[Path] = None
        for parent in [package_root, *package_root.parents]:
            if (parent / ".git").exists():
                git_root = parent
                break
        if git_root is None:
            git_root = package_root.parent

        rel_html = html_root.relative_to(git_root)
        return cls(
            base_url=f"http://localhost:{port}/{git_root.name}/{rel_html}",
            html_root=html_root,
        )
