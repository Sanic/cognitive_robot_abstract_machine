from __future__ import annotations

import operator
import re
from typing import Optional

import inflect

_engine = inflect.engine()

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Call,
    FlatVariable,
    Index,
    MappedVariable,
)
from krrood.entity_query_language.core.variable import (
    InstantiatedVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.operators.aggregators import (
    Average,
    Count,
    CountAll,
    Max,
    Min,
    Mode,
    MultiMode,
    Sum,
)
from krrood.entity_query_language.operators.comparator import Comparator, not_contains
from krrood.entity_query_language.operators.core_logical_operators import AND, OR, Not
from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll
from krrood.entity_query_language.predicate import Predicate
from krrood.entity_query_language.query.operations import (
    GroupedBy,
    Having,
    OrderedBy,
    Where,
)
from krrood.entity_query_language.query.quantifiers import An, ResultQuantifier, The
from krrood.entity_query_language.query.query import Entity, SetOf
from krrood.entity_query_language.verbalization.context import VerbalizationContext, _article

_OP_WORDS = {
    operator.eq: "is",
    operator.ne: "is not",
    operator.lt: "is less than",
    operator.le: "is at most",
    operator.gt: "is greater than",
    operator.ge: "is at least",
    operator.contains: "contains",
    not_contains: "does not contain",
}

_OP_WORDS_COMPACT = {
    operator.eq: "equals",
    operator.ne: "does not equal",
    operator.lt: "less than",
    operator.le: "at most",
    operator.gt: "greater than",
    operator.ge: "at least",
    operator.contains: "contains",
    not_contains: "does not contain",
}

_NEGATED_OP_WORDS = {
    operator.gt: "is not greater than",
    operator.lt: "is not less than",
    operator.ge: "is not at least",
    operator.le: "is not at most",
    operator.eq: "is not",
    operator.ne: "is",
    operator.contains: "does not contain",
    not_contains: "contains",
}

_NEGATED_OP_WORDS_COMPACT = {
    operator.gt: "not greater than",
    operator.lt: "not less than",
    operator.ge: "not at least",
    operator.le: "not at most",
    operator.eq: "does not equal",
    operator.ne: "equals",
    operator.contains: "does not contain",
    not_contains: "contains",
}


def _camel_to_words(name: str) -> str:
    """Convert a CamelCase class name to space-separated lowercase words.

    Examples: ``"HasRole"`` → ``"has role"``, ``"IsReachable"`` → ``"is reachable"``.
    """
    return re.sub(r"([A-Z])", r" \1", name).strip().lower()

def _ordinal(n: int) -> str:
    return _engine.ordinal(_engine.number_to_words(n + 1))


def _ensure_plural(word: str) -> str:
    """Return *word* in plural form, without double-pluralising already-plural words."""
    return word if _engine.singular_noun(word) else _engine.plural(word)


def _plural_possessive(word: str) -> str:
    """Return the plural-possessive form: 'Cabinets'' or 'Children's'."""
    plural = _engine.plural(word)
    return plural + "'" if plural.endswith("s") else plural + "'s"


class EQLVerbalizer:
    """
    Visitor-based verbalizer: maps an EQL expression tree to readable English.

    Usage::

        verbalizer = EQLVerbalizer()
        text = verbalizer.verbalize(query)

    Each ``_v_<ClassName>_`` method handles one node type.  Unknown types fall
    back to :meth:`_v_default_` which returns the node's ``_name_`` property.
    """

    # ── Dispatcher ─────────────────────────────────────────────────────────────

    def verbalize(
        self,
        expr: SymbolicExpression,
        ctx: Optional[VerbalizationContext] = None,
    ) -> str:
        if ctx is None:
            ctx = VerbalizationContext.from_expression(expr)
        method = getattr(self, f"_v_{type(expr).__name__}_", self._v_default_)
        return method(expr, ctx)

    # ── Leaves ─────────────────────────────────────────────────────────────────

    def _v_Variable_(self, expr: Variable, ctx: VerbalizationContext) -> str:
        return ctx.noun_for(expr)

    def _v_Literal_(self, expr: Literal, ctx: VerbalizationContext) -> str:
        return ctx.type_name_of_value(expr._value_)

    def _v_ExternallySetVariable_(self, expr, ctx: VerbalizationContext) -> str:
        type_name = expr._type_.__name__ if getattr(expr, "_type_", None) else "variable"
        return f"{_article(type_name)} {type_name}"

    # ── MappedVariables ────────────────────────────────────────────────────────

    def _v_Attribute_(self, expr: Attribute, ctx: VerbalizationContext) -> str:
        return self._verbalize_mapped_chain_(expr, ctx)

    def _v_Index_(self, expr: Index, ctx: VerbalizationContext) -> str:
        return self._verbalize_mapped_chain_(expr, ctx)

    def _v_Call_(self, expr: Call, ctx: VerbalizationContext) -> str:
        return self._verbalize_mapped_chain_(expr, ctx)

    def _v_FlatVariable_(self, expr: FlatVariable, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr._child_, ctx)

    def _verbalize_plural_(self, expr, ctx: VerbalizationContext) -> str:
        """
        Return a plural noun phrase for *expr* — used by ForAll and aggregators.

        * ``Variable(T)``              → ``"Employees"``
        * ``FlatVariable(child)``      → recurse into child
        * ``Attribute`` (single-hop)   → ``"Cabinets' containers"``
        * anything else                → fall back to singular ``verbalize()``
        """
        if isinstance(expr, FlatVariable):
            return self._verbalize_plural_(expr._child_, ctx)

        if isinstance(expr, Variable):
            type_name = expr._type_.__name__
            label = ctx.disambiguation_map.get(expr._id_, type_name)
            ctx.seen[expr._id_] = label
            if label != type_name:
                return label
            return _engine.plural(type_name)

        if isinstance(expr, Attribute):
            # Walk the chain to find root variable and single attribute hop.
            chain: list = []
            current = expr
            while isinstance(current, MappedVariable):
                chain.append(current)
                current = current._child_
            root = current
            if isinstance(root, Variable) and len(chain) == 1 and isinstance(chain[0], Attribute):
                type_name = root._type_.__name__
                label = ctx.disambiguation_map.get(root._id_, type_name)
                ctx.seen[root._id_] = label
                if label != type_name:
                    root_possessive = f"{label}'s"
                else:
                    root_possessive = _plural_possessive(type_name)
                attr_plural = _ensure_plural(chain[0]._attribute_name_)
                return f"{root_possessive} {attr_plural}"

        return self.verbalize(expr, ctx)

    def _verbalize_mapped_chain_(self, expr: MappedVariable, ctx: VerbalizationContext,
                                 negated: bool = False) -> str:
        """
        Natural-language path for a MappedVariable chain.

        * Boolean terminal ``Attribute``: predicative —
          ``"Robot's battery is [not] active"``.
        * Single-hop non-boolean ``Attribute`` on a root: possessive —
          ``"Robot's battery"``.
        * Longer or mixed chains: ``"of"`` form —
          ``"name of tasks[0] of the Robot"``.

        When the chain root is an ``Entity`` (a sub-query), it is rendered as a
        bare noun phrase (``"a FixedConnection"``) and its where-conditions are
        deferred to the current constraint frame so the enclosing
        ``InstantiatedVariable`` can emit them as a ``"such that …"`` clause.
        """
        chain: list[MappedVariable] = []
        current = expr
        while isinstance(current, MappedVariable):
            chain.append(current)
            current = current._child_
        # _update_children_ replaces a freshly-built Entity with its An/The wrapper;
        # unwrap any ResultQuantifier layers to recover the underlying Entity.
        inner = current
        while isinstance(inner, ResultQuantifier):
            inner = inner._child_
        if isinstance(inner, Entity):
            root_text = self._verbalize_entity_as_inline_noun_(inner, ctx)
        else:
            root_text = self.verbalize(current, ctx)
        chain.reverse()  # root-side first

        terminal = chain[-1]
        if isinstance(terminal, Attribute) and terminal._type_ is bool:
            nav_text = self._verbalize_navigation_chain_(chain[:-1], root_text)
            verb = "is not" if negated else "is"
            return f"{nav_text} {verb} {terminal._attribute_name_}"

        path_parts = self._build_path_parts_(chain)
        if len(path_parts) == 1 and isinstance(expr, Attribute):
            return f"{root_text}'s {path_parts[0]}"
        return " of ".join(reversed(path_parts)) + f" of {root_text}"

    def _verbalize_navigation_chain_(self, nav_chain: list, root_text: str) -> str:
        """
        Verbalize the navigation portion of a chain (everything before a boolean terminal).

        An integer ``Index`` at the end of the chain is converted to an ordinal:
        ``[Attribute("tasks"), Index(0)]`` → ``"the first of the Robot's tasks"``.
        """
        if not nav_chain:
            return root_text

        if isinstance(nav_chain[-1], Index) and isinstance(nav_chain[-1]._key_, int):
            ordinal = _ordinal(nav_chain[-1]._key_)
            pre_parts = self._build_path_parts_(nav_chain[:-1])
            if pre_parts:
                if len(pre_parts) == 1:
                    pre_text = f"{root_text}'s {pre_parts[0]}"
                else:
                    pre_text = " of ".join(reversed(pre_parts)) + f" of {root_text}"
            else:
                pre_text = root_text
            return f"the {ordinal} of {pre_text}"

        path_parts = self._build_path_parts_(nav_chain)
        if len(path_parts) == 1:
            return f"{root_text}'s {path_parts[0]}"
        return " of ".join(reversed(path_parts)) + f" of {root_text}"

    def _build_path_parts_(self, chain: list) -> list[str]:
        """Build readable string fragments for a root-to-leaf MappedVariable chain."""
        parts: list[str] = []
        i = 0
        while i < len(chain):
            node = chain[i]
            if isinstance(node, Attribute):
                name = node._attribute_name_
                # Eagerly absorb immediately following Index nodes into the attr name.
                while i + 1 < len(chain) and isinstance(chain[i + 1], Index):
                    i += 1
                    name += f"[{repr(chain[i]._key_)}]"
                parts.append(name)
            elif isinstance(node, Index):
                parts.append(f"[{repr(node._key_)}]")
            elif isinstance(node, Call):
                parts.append("()")
            elif isinstance(node, FlatVariable):
                pass  # FlatVariable handled by _v_FlatVariable_
            i += 1
        return parts

    # ── Instantiated (predicates / inference variables) ────────────────────────

    def _v_InstantiatedVariable_(
        self, expr: InstantiatedVariable, ctx: VerbalizationContext
    ) -> str:
        # ── 1. Template takes priority (Predicates with _verbalization_template_) ──
        template: Optional[str] = getattr(expr._type_, "_verbalization_template_", None)
        if template is not None:
            kwargs = {
                name: (
                    ctx.type_name_of_value(child._value_)
                    if isinstance(child, Literal)
                    else self.verbalize(child, ctx)
                )
                for name, child in expr._child_vars_.items()
            }
            return template.format(**kwargs)

        type_name = getattr(expr._type_, "__name__", str(expr._type_))

        # ── 2. Predicate subclasses without a template ─────────────────────────────
        if isinstance(expr._type_, type) and issubclass(expr._type_, Predicate):
            if len(expr._child_vars_) == 2:
                # Binary predicate → subject predicate-words object triple form.
                items = list(expr._child_vars_.items())
                left = items[0][1]
                right = items[1][1]
                predicate_text = _camel_to_words(type_name)
                left_text = (
                    ctx.type_name_of_value(left._value_)
                    if isinstance(left, Literal)
                    else self.verbalize(left, ctx)
                )
                right_text = (
                    ctx.type_name_of_value(right._value_)
                    if isinstance(right, Literal)
                    else self.verbalize(right, ctx)
                )
                return f"{left_text} {predicate_text} {right_text}"
            # Other arities → generic constructor-like fallback.
            if expr._child_vars_:
                args_str = ", ".join(
                    f"{name}={ctx.type_name_of_value(child._value_) if isinstance(child, Literal) else self.verbalize(child, ctx)}"
                    for name, child in expr._child_vars_.items()
                )
                return f"{_article(type_name)} {type_name}({args_str})"
            return f"{_article(type_name)} {type_name}"

        # ── 3. Non-predicate class → natural English with binding + deferred clauses ─
        if expr._id_ in ctx.seen:
            return f"the {ctx.seen[expr._id_]}"
        ctx.seen[expr._id_] = type_name

        ctx.push_constraint_frame()

        binding_parts: list[str] = []
        for field_name, child_expr in expr._child_vars_.items():
            value_text = (
                ctx.type_name_of_value(child_expr._value_)
                if isinstance(child_expr, Literal)
                else self.verbalize(child_expr, ctx)
            )
            binding_parts.append(f"the {type_name}'s {field_name} is {value_text}")

        constraints = ctx.pop_constraint_frame()

        result = f"{_article(type_name)} {type_name}"
        if binding_parts:
            result += ", where " + " and ".join(binding_parts)
        if constraints:
            result += ", such that " + " and ".join(constraints)
        return result

    # ── Logical operators ──────────────────────────────────────────────────────

    def _v_AND_(self, expr: AND, ctx: VerbalizationContext) -> str:
        parts = [self.verbalize(c, ctx) for c in ctx.flatten_same_type(expr, AND)]
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    def _v_OR_(self, expr: OR, ctx: VerbalizationContext) -> str:
        parts = [self.verbalize(c, ctx) for c in ctx.flatten_same_type(expr, OR)]
        if len(parts) == 1:
            return parts[0]
        return "either " + ", ".join(parts[:-1]) + f", or {parts[-1]}"

    def _v_Not_(self, expr: Not, ctx: VerbalizationContext) -> str:
        child = expr._child_
        # Case 1: negate a comparator — inline the negated verb word.
        if isinstance(child, Comparator):
            left = self.verbalize(child.left, ctx)
            right = self.verbalize(child.right, ctx)
            neg_table = _NEGATED_OP_WORDS_COMPACT if ctx.compact_predicates else _NEGATED_OP_WORDS
            fallback_table = _OP_WORDS_COMPACT if ctx.compact_predicates else _OP_WORDS
            op_word = neg_table.get(
                child.operation, f"not {fallback_table.get(child.operation, child._name_)}"
            )
            return f"{left} {op_word} {right}"
        # Case 2: negate a boolean attribute chain — inline "is not".
        if isinstance(child, MappedVariable):
            # Walk to the terminal to check if it's a boolean Attribute.
            node = child
            while isinstance(node, MappedVariable):
                node = node._child_
            # node is now root; walk chain list to find terminal
            chain = []
            cur = child
            while isinstance(cur, MappedVariable):
                chain.append(cur)
                cur = cur._child_
            chain.reverse()
            if isinstance(chain[-1], Attribute) and chain[-1]._type_ is bool:
                return self._verbalize_mapped_chain_(child, ctx, negated=True)
        # Case 3: fallback — wrap with "not (…)".
        return f"not ({self.verbalize(child, ctx)})"

    # ── Quantifiers ────────────────────────────────────────────────────────────

    def _v_ForAll_(self, expr: ForAll, ctx: VerbalizationContext) -> str:
        var_text = self._verbalize_plural_(expr.variable, ctx)
        cond_text = self.verbalize(expr.condition, ctx)
        return f"for all {var_text}, {cond_text}"

    def _v_Exists_(self, expr: Exists, ctx: VerbalizationContext) -> str:
        var_text = self.verbalize(expr.variable, ctx)
        cond_text = self.verbalize(expr.condition, ctx)
        return f"there exists {var_text} such that {cond_text}"

    # ── Comparators ────────────────────────────────────────────────────────────

    def _v_Comparator_(self, expr: Comparator, ctx: VerbalizationContext) -> str:
        left = self.verbalize(expr.left, ctx)
        right = self.verbalize(expr.right, ctx)
        table = _OP_WORDS_COMPACT if ctx.compact_predicates else _OP_WORDS
        op_word = table.get(expr.operation, expr._name_)
        return f"{left} {op_word} {right}"

    # ── Aggregators ────────────────────────────────────────────────────────────

    def _v_Count_(self, expr: Count, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "count of {}")

    def _v_CountAll_(self, expr: CountAll, ctx: VerbalizationContext) -> str:
        return "count of all"

    def _v_Sum_(self, expr: Sum, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "sum of {}")

    def _v_Average_(self, expr: Average, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "average of {}")

    def _v_Max_(self, expr: Max, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "maximum {}")

    def _v_Min_(self, expr: Min, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "minimum {}")

    def _v_Mode_(self, expr: Mode, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "mode of {}")

    def _v_MultiMode_(self, expr: MultiMode, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "all modes of {}")

    def _verbalize_aggregator_(self, expr, ctx: VerbalizationContext, template: str) -> str:
        """
        Verbalize an aggregator with coreference: first mention returns the plain phrase;
        any subsequent mention of the same expression prefixes it with "the".
        """
        child_text = self._verbalize_plural_(expr._child_, ctx)
        phrase = template.format(child_text)
        if expr._id_ in ctx.seen:
            return f"the {phrase}"
        ctx.seen[expr._id_] = phrase
        return phrase

    # ── Query: Entity and SetOf ────────────────────────────────────────────────

    def _v_Entity_(self, expr: Entity, ctx: VerbalizationContext) -> str:
        if expr._id_ in ctx.seen:
            return f"the {ctx.seen[expr._id_]}"

        expr.build()
        is_the = (
            expr._quantifier_builder_ is not None
            and expr._quantifier_builder_.type is The
        )
        var = expr.selected_variable

        if isinstance(var, Entity):
            selected = self._verbalize_entity_as_noun_(var, ctx)
        elif var is None:
            selected_type = "entity"
            ctx.seen[expr._id_] = selected_type
            selected = "entities"
        elif is_the:
            selected_type = var._type_.__name__ if getattr(var, "_type_", None) else "entity"
            ctx.seen[var._id_] = selected_type
            ctx.seen[expr._id_] = selected_type
            selected = f"the unique {selected_type}"
        else:
            selected = self.verbalize(var, ctx)
            selected_type = ctx.seen.get(getattr(var, "_id_", None), "entity")
            ctx.seen[expr._id_] = selected_type

        return self._verbalize_query_body_(expr, ctx, f"Find {selected}")

    def _verbalize_entity_as_noun_(self, expr: Entity, ctx: VerbalizationContext) -> str:
        """
        Compact form used when an ``Entity`` acts as the selected variable of an outer query.

        Produces ``"the unique Container where its name equals …"`` rather than
        ``"Find the unique Container, such that …"``.
        """
        if expr._id_ in ctx.seen:
            return f"the {ctx.seen[expr._id_]}"

        expr.build()
        is_the = (
            expr._quantifier_builder_ is not None
            and expr._quantifier_builder_.type is The
        )
        var = expr.selected_variable
        selected_type = var._type_.__name__ if var and getattr(var, "_type_", None) else "entity"

        ctx.seen[expr._id_] = selected_type
        if var is not None:
            ctx.seen[var._id_] = selected_type

        if is_the:
            article_noun = f"the unique {selected_type}"
        else:
            article_noun = f"{_article(selected_type)} {selected_type}"

        where_expr = expr._where_expression_
        if where_expr is not None:
            cond = self.verbalize(where_expr.condition, ctx)
            return f"{article_noun} where {cond}"
        return article_noun

    def _verbalize_entity_as_inline_noun_(self, entity: Entity, ctx: VerbalizationContext) -> str:
        """
        Render an ``Entity`` as a bare noun phrase for use as the root of an
        ``Attribute`` chain inside an ``InstantiatedVariable``.

        On the **first** encounter the entity's type name is registered in
        *ctx.seen* (so subsequent mentions use "the") and its where-conditions
        are deferred into the current constraint frame.  On **subsequent**
        encounters the method returns ``"the <TypeName>"`` immediately with no
        side effects.
        """
        if entity._id_ in ctx.seen:
            return f"the {ctx.seen[entity._id_]}"

        entity.build()
        var = entity.selected_variable
        type_name = var._type_.__name__ if var and getattr(var, "_type_", None) else "entity"

        ctx.seen[entity._id_] = type_name
        if var is not None and hasattr(var, "_id_"):
            ctx.seen[var._id_] = type_name

        where_expr = entity._where_expression_
        if where_expr is not None:
            cond_text = self.verbalize(where_expr.condition, ctx)
            ctx.add_constraint(cond_text)

        return f"{_article(type_name)} {type_name}"

    def _v_SetOf_(self, expr: SetOf, ctx: VerbalizationContext) -> str:
        expr.build()
        vars_str = ", ".join(self.verbalize(v, ctx) for v in expr._selected_variables_)
        return self._verbalize_query_body_(expr, ctx, f"Find sets of ({vars_str})")

    def _verbalize_query_body_(self, expr, ctx: VerbalizationContext, prefix: str) -> str:
        """Append where / grouped-by / having / ordered-by clauses to *prefix*."""
        parts = [prefix]

        where_expr = expr._where_expression_
        grouped_expr = expr._grouped_by_expression_
        having_expr = expr._having_expression_

        if where_expr is not None:
            parts.append(f"such that {self.verbalize(where_expr.condition, ctx)}")

        if grouped_expr is not None and grouped_expr.variables_to_group_by:
            groups = [self.verbalize(v, ctx) for v in grouped_expr.variables_to_group_by]
            parts.append(f"grouped by {', '.join(groups)}")

        if having_expr is not None:
            ctx.compact_predicates = True
            having_text = self.verbalize(having_expr.condition, ctx)
            ctx.compact_predicates = False
            parts.append(f"having {having_text}")

        ob = expr._ordered_by_builder_
        if ob is not None:
            direction = "descending" if ob.descending else "ascending"
            parts.append(f"ordered by {self.verbalize(ob.variable, ctx)} ({direction})")

        return ", ".join(parts)

    # ── Result quantifiers (transparent wrappers) ──────────────────────────────

    def _v_An_(self, expr: An, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr._child_, ctx)

    def _v_The_(self, expr: The, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr._child_, ctx)

    def _v_ResultQuantifier_(self, expr: ResultQuantifier, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr._child_, ctx)

    # ── Filter wrappers (delegate to their condition) ──────────────────────────

    def _v_Where_(self, expr: Where, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr.condition, ctx)

    def _v_Having_(self, expr: Having, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr.condition, ctx)

    def _v_GroupedBy_(self, expr: GroupedBy, ctx: VerbalizationContext) -> str:
        if expr.variables_to_group_by:
            groups = [self.verbalize(v, ctx) for v in expr.variables_to_group_by]
            return f"grouped by {', '.join(groups)}"
        return "grouped"

    def _v_OrderedBy_(self, expr: OrderedBy, ctx: VerbalizationContext) -> str:
        direction = "descending" if expr.descending else "ascending"
        return f"ordered by {self.verbalize(expr.variable, ctx)} ({direction})"

    # ── Fallback ───────────────────────────────────────────────────────────────

    def _v_default_(self, expr: SymbolicExpression, ctx: VerbalizationContext) -> str:
        return expr._name_
