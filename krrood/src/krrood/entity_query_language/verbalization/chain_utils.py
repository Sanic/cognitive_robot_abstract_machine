"""
Utilities for walking MappedVariable chains, building path parts, and
pluralising chain expressions.

These are pure utilities shared by multiple verbalizer subsystems.  They must
not import from the subsystem files (``verbalizer.py``, ``context.py``,
``rules/*.py``) to avoid circular dependencies.  Imports from EQL core and
from the ``fragments/``, ``vocabulary/`` layers are safe because those are
lower in the dependency graph.
"""
from __future__ import annotations

import datetime as _dt
from typing import Callable, Optional, TYPE_CHECKING

from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Call,
    FlatVariable,
    Index,
    MappedVariable,
)
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef
from krrood.entity_query_language.verbalization.utils import _ensure_plural, inflect_engine
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.fragments.base import VerbFragment


def walk_chain(expression) -> tuple[list, object]:
    """
    Walk a :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
    chain outward-first and return ``(chain, root)``.

    The chain is reversed so index 0 is the outermost (innermost to the call
    site) node; the root is the first non-MappedVariable child.

    Example: for ``robot.arm.joint`` the chain is
    ``[Attribute('joint'), Attribute('arm')]`` and root is the ``robot`` Variable.

    :param expression: Any expression; non-MappedVariable expressions return an
        empty chain with *expression* as the root.
    :returns: Tuple ``(chain, root)`` where *chain* is the core
        :attr:`~krrood.entity_query_language.core.mapped_variable.MappedVariable._access_path_`
        (root-adjacent first, terminal last) and *root* is the chain base.
    :rtype: tuple[list, object]
    """
    if isinstance(expression, MappedVariable):
        return list(expression._access_path_), expression._chain_root_
    return [], expression


def is_temporal(expression) -> bool:
    """
    Return ``True`` when *expression* denotes a :class:`datetime.datetime` value or variable.

    Used by comparator verbalization to select temporal operator phrases
    (*"is before"* / *"is after"*) instead of relational ones.  Inspects the
    expression's ``_type_`` (or a :class:`~krrood.entity_query_language.core.variable.Literal`'s
    value), so it is a pure structural/type check with no verbalization state.

    :param expression: Any EQL expression.
    :returns: ``True`` when the expression is datetime-typed.
    :rtype: bool
    """
    if isinstance(expression, Literal):
        return isinstance(expression._value_, _dt.datetime)
    if isinstance(expression, Variable):
        return getattr(expression, "_type_", None) is _dt.datetime
    if isinstance(expression, MappedVariable):
        chain, _ = walk_chain(expression)
        return bool(chain) and getattr(chain[-1], "_type_", None) is _dt.datetime
    return False


def chain_root(expression) -> object:
    """
    Return the non-:class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
    root of *expression* without building the full chain list.

    Faster than :func:`walk_chain` when only the root is needed.

    :param expression: Any expression.
    :returns: The deepest non-MappedVariable node in the chain, or *expression* itself
        when it is not a MappedVariable.
    :rtype: object
    """
    return expression._chain_root_ if isinstance(expression, MappedVariable) else expression


def build_path_parts(chain: list) -> list[tuple[str, Optional[SourceRef]]]:
    """
    Convert a walked chain (from :func:`walk_chain`) into ``(display_name, SourceRef | None)`` pairs.

    Merging rules:

    * Consecutive ``Attribute → Index`` pairs are merged into ``"attr[key]"`` with ``ref=None``
      (composite indexed access has no clean single-symbol anchor).
    * Standalone :class:`~krrood.entity_query_language.core.mapped_variable.Index` nodes
      appear as ``"[key]"`` with ``ref=None``.
    * :class:`~krrood.entity_query_language.core.mapped_variable.Call` nodes appear as ``"()"``
      with ``ref=None``.
    * :class:`~krrood.entity_query_language.core.mapped_variable.FlatVariable` nodes are skipped.

    :param chain: Outermost-first chain list from :func:`walk_chain`.
    :type chain: list
    :returns: Ordered list of ``(display_name, SourceRef | None)`` pairs,
        outermost attribute first.
    :rtype: list[tuple[str, SourceRef | None]]
    """
    parts: list[tuple[str, Optional[SourceRef]]] = []
    i = 0
    while i < len(chain):
        node = chain[i]
        if isinstance(node, Attribute):
            name = node._attribute_name_
            owner = node._owner_class_
            ref: Optional[SourceRef] = SourceRef.for_attribute(owner, name)
            while i + 1 < len(chain) and isinstance(chain[i + 1], Index):
                i += 1
                name += f"[{repr(chain[i]._key_)}]"
                ref = None  # composite indexed access has no clean single-line anchor
            parts.append((name, ref))
        elif isinstance(node, Index):
            parts.append((f"[{repr(node._key_)}]", None))
        elif isinstance(node, Call):
            parts.append(("()", None))
        elif isinstance(node, FlatVariable):
            pass
        i += 1
    return parts


def verbalize_plural(expression, context: VerbalizationContext, build_fn: Callable) -> VerbFragment:
    """
    Return a plural :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
    for *expression*.

    Handles three special cases with dedicated plural forms:

    * :class:`~krrood.entity_query_language.core.mapped_variable.FlatVariable` — delegates to its child.
    * :class:`~krrood.entity_query_language.core.variable.Variable` — pluralises the type name
      (e.g. ``Robot`` → ``Robots``).
    * Single :class:`~krrood.entity_query_language.core.mapped_variable.Attribute` on a Variable —
      produces *"attrs of Roots"*.

    Falls back to *build_fn* for all other expression types.

    :param expression: EQL expression to pluralise.
    :param context: Shared verbalization state.
    :type context: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
    :param build_fn: The top-level dispatcher (``EQLVerbalizer.build``) used as fallback.
    :type build_fn: Callable
    :returns: Plural fragment for *expression*.
    :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
    """
    if isinstance(expression, FlatVariable):
        return verbalize_plural(expression._child_, context, build_fn)

    if isinstance(expression, Variable):
        type_name = expression._type_.__name__
        label = context.disambiguation_map.get(expression._id_, type_name)
        context.seen[expression._id_] = label
        plural = label if label != type_name else inflect_engine.plural(type_name)
        return RoleFragment.for_variable(plural, expression)

    if isinstance(expression, Attribute):
        chain, root = walk_chain(expression)
        if isinstance(root, Variable) and len(chain) == 1 and isinstance(chain[0], Attribute):
            type_name = root._type_.__name__
            label = context.disambiguation_map.get(root._id_, type_name)
            context.seen[root._id_] = label
            root_plural = label if label != type_name else inflect_engine.plural(type_name)
            attribute_name = chain[0]._attribute_name_
            owner = chain[0]._owner_class_
            return PhraseFragment(
                parts=[
                    RoleFragment.for_attribute(owner, attribute_name, plural=True),
                    Prepositions.OF.as_fragment(),
                    RoleFragment.for_variable(root_plural, root),
                ],
                separator=" ",
            )

    return build_fn(expression, context)
