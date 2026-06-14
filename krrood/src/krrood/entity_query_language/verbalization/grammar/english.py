from __future__ import annotations

import sys

from typing_extensions import List

from krrood.entity_query_language.core.mapped_variable import (
    FlatVariable,
    MappedVariable,
)
from krrood.entity_query_language.core.variable import (
    ExternallySetVariable,
    InstantiatedVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    OR,
    Not,
    flatten_operands,
)
from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    oxford_comma,
    PhraseFragment,
    RoleFragment,
    Fragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
    Separator,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_boolean_attribute_chain,
)
from krrood.entity_query_language.verbalization.grammar.conditions.verbalizer import (
    ConditionVerbalizer,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
    PhraseRule,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    fold_range_pairs,
    fragment_for_folded_conjunct,
    has_pair,
)
from krrood.entity_query_language.verbalization.rendering.realization import (
    realize_subtree,
)
from krrood.entity_query_language.verbalization.grammar.assembly.chains import (
    ChainAssembler,
)
from krrood.entity_query_language.verbalization.grammar.planning.chains import (
    ChainPlanner,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    FallbackNouns,
    Keywords,
    Logicals,
    Prepositions,
    Punctuation,
)
from krrood.entity_query_language.verbalization.vocabulary.words import ChildForm
from krrood.ormatic.utils import classes_of_module

from krrood.entity_query_language.core.base_expressions import Filter
from krrood.entity_query_language.query.operations import GroupedBy, OrderedBy
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity, SetOf
from krrood.entity_query_language.verbalization.grammar.aggregation_kinds import (
    AGGREGATION_KIND,
)
from krrood.entity_query_language.verbalization.grammar.assembly.inference import (
    InferenceAssembler,
)
from krrood.entity_query_language.verbalization.grammar.assembly.clauses import (
    GroupedByAssembler,
    OrderedByAssembler,
)
from krrood.entity_query_language.verbalization.grammar.assembly.query import (
    QueryAssembler,
)
from krrood.entity_query_language.verbalization.grammar.planning.inference import (
    InferencePlanner,
)
from krrood.entity_query_language.verbalization.grammar.assembly.instantiated import (
    InstantiatedAssembler,
)
from krrood.entity_query_language.verbalization.grammar.planning.instantiated import (
    InstantiatedPlanner,
)

# ── comparator ───────────────────────────────────────────────────────────────


class ComparatorRule(PhraseRule):
    """``<left> <operator> <right>`` — e.g. *"is greater than 50"*.

    >>> verbalize_expression(variable(Robot, []).battery > 50)
    'the battery of a Robot is greater than 50'
    """

    construct = Comparator
    name = "comparator"

    def build(self, node: Comparator, context: RuleContext) -> Fragment:
        return ConditionVerbalizer(context).predicate(node)


# ── variables ──────────────────────────────────────────────────────────────────


class VariableRule(PhraseRule):
    """*"a/an Robot"* (first mention), *"the Robot"* (subsequent), or *"Robot N"* (numbered).

    >>> verbalize_expression(variable(Robot, []))
    'a Robot'
    """

    construct = Variable
    name = "variable"

    def build(self, node: Variable, context: RuleContext) -> Fragment:
        if context.number is Number.PLURAL:
            return self._plural(node, context)
        noun_form = context.refer.noun_for_parts(node)
        return NounPhrase(
            head=RoleFragment.for_variable(noun_form.label, node),
            definiteness=noun_form.definiteness,
            referent_id=node._id_,
        )

    @staticmethod
    def _plural(node: Variable, context: RuleContext) -> Fragment:
        """Bare plural variable noun phrase (*"Robots"*); the determiner phase drops the article and
        the morphology pass inflects the head.

        A numbered label (*"Robot 2"*) is surface-final — kept singular and bare; a plain type
        name is a plural indefinite noun phrase (the concord table renders it bare-then-pluralised).
        """
        numbered = context.refer.numbered_label(node)
        return NounPhrase(
            head=RoleFragment.for_variable(numbered.text, node),
            number=Number.SINGULAR if numbered.is_numbered else Number.PLURAL,
            definiteness=(
                Definiteness.BARE if numbered.is_numbered else Definiteness.INDEFINITE
            ),
            referent_id=node._id_,
        )


class LiteralRule(PhraseRule):
    """A literal value (e.g. ``42``, ``"hello"``, ``True``)."""

    construct = Literal
    name = "literal"

    def build(self, node: Literal, context: RuleContext) -> Fragment:
        return RoleFragment(
            text=context.services.type_name_of_value(node._value_),
            role=SemanticRole.LITERAL,
        )


class ExternalVariableRule(PhraseRule):
    """*"a/an TypeName"* for an opaque externally-set variable (no coreference)."""

    construct = ExternallySetVariable
    name = "external-variable"

    def build(self, node: ExternallySetVariable, context: RuleContext) -> Fragment:
        type_name = (
            node._type_.__name__
            if getattr(node, "_type_", None)
            else FallbackNouns.VARIABLE.text
        )
        return NounPhrase(head=RoleFragment(text=type_name, role=SemanticRole.VARIABLE))


# ── logical ──────────────────────────────────────────────────────────────────


class AndRule(PhraseRule):
    """Conjunction *"a, b, and c"* (Oxford comma); flattens nested ANDs.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(and_(robot.battery > 50, robot.name == 'x'))
    "the battery of a Robot is greater than 50, and the name of the Robot is 'x'"
    """

    construct = AND
    name = "and"

    def build(self, node: AND, context: RuleContext) -> Fragment:
        parts = [context.child(conjunct) for conjunct in flatten_operands(node, AND)]
        if len(parts) == 1:
            return parts[0]
        return oxford_comma(parts, Conjunctions.AND.as_fragment())


class RangeConjunctionRule(PhraseRule):
    """A conjunction containing a low/high pair on one chain → *"… is between low and high"*.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(and_(robot.battery > 10, robot.battery < 90))
    'the battery of a Robot is between 10, and 90'
    """

    construct = AND
    name = "and-range"

    def when(self, node: AND, context: RuleContext) -> bool:
        return has_pair(flatten_operands(node, AND))

    def build(self, node: AND, context: RuleContext) -> Fragment:
        parts: List[Fragment] = [
            fragment_for_folded_conjunct(
                item, context.child, compact=context.configuration.compact_predicates
            )
            for item in fold_range_pairs(flatten_operands(node, AND))
        ]
        if len(parts) == 1:
            return parts[0]
        return oxford_comma(parts, Conjunctions.AND.as_fragment())


class OrRule(PhraseRule):
    """Disjunction *"either a, b, or c"*; flattens nested ORs.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(or_(robot.battery > 50, robot.battery < 10))
    'either the battery of a Robot is greater than 50, or the battery of the Robot is less than 10'
    """

    construct = OR
    name = "or"

    def build(self, node: OR, context: RuleContext) -> Fragment:
        parts = [context.child(conjunct) for conjunct in flatten_operands(node, OR)]
        if len(parts) == 1:
            return parts[0]
        # "either a, b, or c": the head items are comma-joined, then a trailing comma that the
        # orthography pass hugs to the last head item — no separator="" bookkeeping here.
        head = PhraseFragment(parts=parts[:-1], separator=Separator.COMMA)
        return PhraseFragment(
            parts=[
                Logicals.EITHER.as_fragment(),
                head,
                Punctuation.COMMA.as_fragment(),
                Conjunctions.OR.as_fragment(),
                parts[-1],
            ]
        )


class NotRule(PhraseRule):
    """Generic negation *"not (<child>)"* (specialised by the guarded Not rules below).

    >>> verbalize_expression(Not(IsReachable(variable(Location, []))))
    'not (a Location is reachable)'
    """

    construct = Not
    name = "not"

    def build(self, node: Not, context: RuleContext) -> Fragment:
        child_fragment = context.child(node._child_)
        # The parens glue to the child via the orthography pass → "not (child)".
        return PhraseFragment(
            parts=[
                Logicals.NOT.as_fragment(),
                Punctuation.OPEN_PAREN.as_fragment(),
                child_fragment,
                Punctuation.CLOSE_PAREN.as_fragment(),
            ]
        )


class NotComparatorRule(PhraseRule):
    """Inline negated comparator *"a is not greater than b"* (Not over a Comparator).

    >>> verbalize_expression(Not(variable(Robot, []).battery > 50))
    'the battery of a Robot is not greater than 50'
    """

    construct = Not
    name = "not-comparator"

    def when(self, node: Not, context: RuleContext) -> bool:
        return isinstance(node._child_, Comparator)

    def build(self, node: Not, context: RuleContext) -> Fragment:
        return ConditionVerbalizer(context).predicate(node._child_, negated=True)


class NotBooleanAttributeRule(PhraseRule):
    """Negated boolean attribute chain *"<nav> is not <attribute>"* (Not over a bool-attribute chain).

    >>> verbalize_expression(Not(variable(Task, []).completed))
    'a Task is not completed'
    """

    construct = Not
    name = "not-bool-attribute"

    def when(self, node: Not, context: RuleContext) -> bool:
        return is_boolean_attribute_chain(node._child_)

    def build(self, node: Not, context: RuleContext) -> Fragment:
        plan = ChainPlanner(node._child_).plan()
        return ChainAssembler(context).boolean_predicative(plan, negated=True)


# ── chains (MappedVariable) ──────────────────────────────────────────────────
#
# One guarded rule per surface form, dispatched by ``select``. The guards are mutually exclusive,
# so at most one fires and no ordering between them is needed: the precedence "the bare-plural
# noun phrase wins over the predicative" lives in ``ChainPlan.renders_as_plural_attribute``, which
# both the plural and boolean rules consult. The guarded forms outrank the unguarded possessive
# fallback. Adding a chain form is a new guarded rule here — no existing rule changes.


class PluralChainAttributeRule(PhraseRule):
    """Plural single-attribute chain → bare plural *"attributes of Roots"*.

    >>> verbalize_expression(sum(variable(Robot, []).battery))
    'the sum of batteries of Robots'
    """

    construct = MappedVariable
    name = "chain-plural-attribute"

    def when(self, node: MappedVariable, context: RuleContext) -> bool:
        return ChainPlanner(node).plan().renders_as_plural_attribute(context.number)

    def build(self, node: MappedVariable, context: RuleContext) -> Fragment:
        return ChainAssembler(context).plural_attribute(ChainPlanner(node).plan())


class BooleanAttributeChainRule(PhraseRule):
    """Boolean-terminal chain → predicative *"<navigation> is <attribute>"* (unless the bare-plural
    attribute form takes precedence).

    >>> verbalize_expression(variable(Task, []).completed)
    'a Task is completed'
    """

    construct = MappedVariable
    name = "chain-boolean-attribute"

    def when(self, node: MappedVariable, context: RuleContext) -> bool:
        plan = ChainPlanner(node).plan()
        return plan.is_boolean_terminal and not plan.renders_as_plural_attribute(
            context.number
        )

    def build(self, node: MappedVariable, context: RuleContext) -> Fragment:
        return ChainAssembler(context).boolean_predicative(ChainPlanner(node).plan())


class PossessiveChainRule(PhraseRule):
    """Any attribute / index / call chain → possessive path *"the attribute of the Root"*
    (the unguarded fallback form).

    >>> verbalize_expression(variable(Task, []).name)
    'the name of a Task'
    """

    construct = MappedVariable
    name = "chain-possessive"

    def build(self, node: MappedVariable, context: RuleContext) -> Fragment:
        return ChainAssembler(context).possessive(ChainPlanner(node).plan())


class FlatVariableRule(PhraseRule):
    """A transparent SetOf wrapper → unwrap to its child (forwarding the requested number)."""

    construct = FlatVariable
    name = "flat-variable"

    def build(self, node: FlatVariable, context: RuleContext) -> Fragment:
        return context.child(node._child_, number=context.number)


# ── aggregators ──────────────────────────────────────────────────────────────


class AggregatorRule(PhraseRule):
    """*"the <aggregation> <plural child>"* (or *"the <aggregation> of <child>"*).

    >>> verbalize_expression(max(variable(Robot, []).battery))
    'the maximum of the battery of a Robot'
    """

    construct = Aggregator
    name = "aggregator"

    def build(self, node: Aggregator, context: RuleContext) -> Fragment:
        aggregation_kind = AGGREGATION_KIND[type(node)]
        aggregation_word = aggregation_kind.value
        aggregation_fragment = aggregation_kind.as_fragment()

        if aggregation_word.child_form is ChildForm.NONE:
            return aggregation_fragment  # childless aggregate, e.g. "count of all"
        if aggregation_word.child_form == ChildForm.SINGULAR_OF:
            child_fragment = context.child(node._child_)
            modifiers = [Prepositions.OF.as_fragment(), child_fragment]
        else:
            child_fragment = context.child(node._child_, number=Number.PLURAL)
            modifiers = [child_fragment]
        return NounPhrase(
            head=aggregation_fragment,
            definiteness=Definiteness.DEFINITE,
            modifiers=modifiers,
        )


# ── quantifiers ──────────────────────────────────────────────────────────────


class ForAllRule(PhraseRule):
    """*"for all <plural var>, <condition>"*.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(for_all(robot, robot.battery > 0))
    'for all Robots, the battery of the Robot is greater than 0'
    """

    construct = ForAll
    name = "for-all"

    def build(self, node: ForAll, context: RuleContext) -> Fragment:
        variable_fragment = context.child(node.variable, number=Number.PLURAL)
        condition_fragment = context.child(node.condition)
        return PhraseFragment(
            parts=[
                Logicals.FOR_ALL.as_fragment(),
                variable_fragment,
                Punctuation.COMMA.as_fragment(),
                condition_fragment,
            ]
        )


class ExistsRule(PhraseRule):
    """*"there exists <variable> such that <condition>"*.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(exists(robot, robot.battery > 0))
    'there exists a Robot such that the battery of the Robot is greater than 0'
    """

    construct = Exists
    name = "exists"

    def build(self, node: Exists, context: RuleContext) -> Fragment:
        variable_fragment = context.child(node.variable)
        condition_fragment = context.child(node.condition)
        return PhraseFragment(
            parts=[
                Logicals.THERE_EXISTS.as_fragment(),
                variable_fragment,
                Keywords.SUCH_THAT.as_fragment(),
                condition_fragment,
            ]
        )


# ── query / entity family ───────────────────────────────────────────────────
# Each rule wires the query planner (what to say) + assembler (realise it).


class TopLevelEntityRule(PhraseRule):
    """Top-level Entity → imperative *"Find …"* (only at query_depth 0).

    >>> verbalize_expression(an(entity(variable(Robot, []))))
    'Find a Robot'
    """

    construct = Entity
    name = "top-level-entity"
    enters_query_scope = True

    def when(self, node: Entity, context: RuleContext) -> bool:
        return context.configuration.query_depth == 0

    def build(self, node: Entity, context: RuleContext) -> Fragment:
        return QueryAssembler(context).assemble(node)


class NestedEntityRule(PhraseRule):
    """Nested Entity → noun phrase (query_depth > 0); never emits *"Find"*.

    >>> worker = variable(Worker, [])
    >>> verbalize_expression(
    ...     an(entity(worker).where(contains(worker.tasks, an(entity(variable(Task, []))))))
    ... )
    'Find a Worker whose tasks contains a Task'
    """

    construct = Entity
    name = "nested-entity"
    enters_query_scope = True

    def when(self, node: Entity, context: RuleContext) -> bool:
        return context.configuration.query_depth > 0

    def build(self, node: Entity, context: RuleContext) -> Fragment:
        return QueryAssembler(context).assemble_nested(node)


class InferenceRuleRule(TopLevelEntityRule):
    """Top-level inference-rule Entity → ``IF … THEN …`` block.

    A refinement of :class:`TopLevelEntityRule`: it applies exactly when that rule does *and* the
    entity is an inference rule, so ``select`` prefers it (more-derived class) over the plain
    top-level form without any tiebreak. Unlike the plain form it does not enter query scope.
    """

    name = "inference-rule"
    enters_query_scope = False

    def when(self, node: Entity, context: RuleContext) -> bool:
        return super().when(node, context) and InferencePlanner.can_handle(node)

    def build(self, node: Entity, context: RuleContext) -> Fragment:
        return InferenceAssembler(context).assemble(node)


class SetOfRule(PhraseRule):
    """SetOf → *"Find (v1, v2, …) such that …"*.

    >>> verbalize_expression(an(set_of(variable(Robot, []), variable(Task, []))))
    'Find sets of (a Robot, a Task)'
    """

    construct = SetOf
    name = "set-of"
    enters_query_scope = True

    def build(self, node: SetOf, context: RuleContext) -> Fragment:
        return QueryAssembler(context).assemble_set_of(node)


class ResultQuantifierRule(PhraseRule):
    """Transparent wrapper (An / The / …) → delegate to the child."""

    construct = ResultQuantifier
    name = "result-quantifier"

    def build(self, node: ResultQuantifier, context: RuleContext) -> Fragment:
        return context.child(node._child_)


class FilterRule(PhraseRule):
    """Transparent wrapper (Where / Having) → delegate to the condition."""

    construct = Filter
    name = "filter"

    def build(self, node: Filter, context: RuleContext) -> Fragment:
        return context.child(node.condition)


class GroupedByRule(PhraseRule):
    """GroupedBy → *"grouped by <key1>, <key2>, …"* (or *"grouped"* when keyless).

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(
    ...     a(set_of(employee.department, sum(employee.salary)).grouped_by(employee.department))
    ... )
    'Find sets of (the department of an Employee, the sum of salaries of Employees) grouped by the department of the Employee'
    """

    construct = GroupedBy
    name = "grouped-by"

    def build(self, node: GroupedBy, context: RuleContext) -> Fragment:
        return GroupedByAssembler(context).assemble(node)


class OrderedByRule(PhraseRule):
    """OrderedBy → *"ordered by <variable> (ascending|descending)"*.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(a(set_of(employee).ordered_by(employee.salary, descending=True)))
    'Find sets of (an Employee) ordered by the salary of the Employee (descending)'
    """

    construct = OrderedBy
    name = "ordered-by"

    def build(self, node: OrderedBy, context: RuleContext) -> Fragment:
        return OrderedByAssembler(context).assemble(node)


# ── instantiated variable ────────────────────────────────────────────────────


class InstantiatedVariableRule(PhraseRule):
    """*"a TypeName where the field of the TypeName is … such that …"*."""

    construct = InstantiatedVariable
    name = "instantiated-variable"

    def build(self, node: InstantiatedVariable, context: RuleContext) -> Fragment:
        return InstantiatedAssembler(context).assemble(node)


class InstantiatedVerbalizableRule(PhraseRule):
    """An InstantiatedVariable whose type supplies a verbalization template string."""

    construct = InstantiatedVariable
    name = "instantiated-verbalizable"

    def when(self, node: InstantiatedVariable, context: RuleContext) -> bool:
        return InstantiatedPlanner.has_template(node)

    def build(self, node: InstantiatedVariable, context: RuleContext) -> Fragment:
        # An opaque format string: it consumes finalized child text, so it realizes its
        # children locally (morphology pass + flatten) rather than deferring to the global pass.
        template = node._type_._verbalization_template_()
        kwargs = {
            name: realize_subtree(context.child(child))
            for name, child in node._child_vars_.items()
        }
        return WordFragment(text=template.format(**kwargs))


# ── the full grammar ─────────────────────────────────────────────────────────────
# Auto-discovered: one instance of every concrete PhraseRule subclass *defined in this
# module* (``classes_of_module`` excludes the imported ``PhraseRule`` base).  A new rule is
# registered simply by defining its class here.  Order is irrelevant — ``select`` decides
# specificity — so the discovery order (alphabetical) is fine.
RULES: List[PhraseRule] = [
    rule_cls()
    for rule_cls in classes_of_module(sys.modules[__name__])
    if issubclass(rule_cls, PhraseRule)
]
