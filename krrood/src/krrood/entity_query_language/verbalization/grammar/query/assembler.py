from __future__ import annotations

import uuid
from dataclasses import replace

from typing_extensions import List, Optional, Tuple

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.query.query import Query, SetOf
from krrood.entity_query_language.verbalization.navigation_path import PathStep
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
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
from krrood.entity_query_language.verbalization.grammar.aggregation.assembler import (
    AggregationValueAssembler,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.chain.planner import (
    ChainPlanner,
)
from krrood.entity_query_language.verbalization.grammar.clauses.composer import (
    ClauseComposer,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    QueryPlan,
    QueryPlanner,
    RankingKeyRelation,
    ReportKind,
    ReportPlan,
    SelectionKind,
)
from krrood.entity_query_language.verbalization.grammar.query.ranking import (
    ranking_surface,
    RankingRequest,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    group_consecutive_by_owner,
    OwnerGroup,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    attribute_fragment,
    coordinated_genitive,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    FallbackNouns,
    GroupingPhrases,
    Keywords,
    Punctuation,
)


def _subject_id(variable: SymbolicExpression) -> Optional[uuid.UUID]:
    """:return: The referent id for a subject variable (``None`` when *variable* is not a single
    variable, which suppresses pronominalisation)."""
    return variable._id_ if isinstance(variable, Variable) else None


class QueryAssembler(Assembler[Query, QueryPlan]):
    """
    Realise a query / nested-entity / set-of from its query plan into the *"Find … such that …
    grouped by … having … ordered by …"* block (and the nested noun-phrase form).

    It dispatches on the selection shape, builds the selection and its restrictions, and combines
    the trailing clauses. Coreference is resolved later: the assembler emits referring noun
    phrases and subject-scope markers, and a document-order pass decides first/subsequent/pronoun
    afterwards (Reiter & Dale 2000).

    Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
    'Find a Robot whose battery is greater than 50'
    """

    planner = QueryPlanner

    # ── entry points ─────────────────────────────────────────────────────────

    def realize(self, node: Query, plan: QueryPlan) -> Fragment:
        """
        :param node: The query being rendered.
        :param plan: The query plan.
        :return: The top-level imperative form *"Find X such that …"*, dispatched on the
            selection shape.
        """
        if plan.report is not None and plan.report.kind is ReportKind.GROUPING:
            return self._assemble_grouped_report(node, plan, plan.report)
        handlers = {
            SelectionKind.ENTITY_SELECTOR: self._realize_entity_selector,
            SelectionKind.EMPTY: self._realize_empty,
            SelectionKind.SUBJECT: self._assemble_subject,
        }
        return handlers[plan.kind](node, plan)

    def _realize_entity_selector(self, node: Query, plan: QueryPlan) -> Fragment:
        """:return: *"Find <a Robot where …> such that …"* — the selected variable is itself an entity."""
        selection = self._as_noun(node.selected_variable)
        return self._query_body(
            node, plan, selection, where_items=[self._where_clause(plan)]
        )

    def _realize_empty(self, node: Query, plan: QueryPlan) -> Fragment:
        """:return: *"Find entities such that …"* — no selected variable (the fallback form)."""
        return self._query_body(
            node,
            plan,
            FallbackNouns.ENTITY.plural_fragment(),
            where_items=[self._where_clause(plan)],
        )

    def assemble_nested(self, node: Query) -> Fragment:
        """
        :param node: The nested entity.
        :return: The noun-phrase form for a nested entity (never emits *"Find …"*).
        """
        plan = self.plan(node)
        if plan.is_aggregation_subquery:
            return AggregationValueAssembler(self.context).realize(node, plan)
        return self._as_noun(node)

    def assemble_set_of(self, node: SetOf) -> Fragment:
        """
        :param node: The set-of query.
        :return: *"Find v1 and v2 such that …"* for a search; *"Report <columns>"* /
            *"For each <keys>, report <columns>"* for an aggregation report; *"Report v1 and v2
            ordered by …"* (plural) for an ordered listing.
        """
        plan = self.plan(node)
        report = plan.report
        if report is not None and report.kind is ReportKind.GROUPING:
            return self._assemble_grouped_report(node, plan, report)
        if report is not None and report.kind is ReportKind.AGGREGATION:
            return self._assemble_aggregation_report(node, plan, report)
        if plan.ranking is not None:
            return self._assemble_ranked_set_of(node, plan)
        return self._query_body(
            node,
            plan,
            self._set_of_selection(node, plan),
            where_items=[self._where_clause(plan)],
            find_header=self._set_of_header(plan),
        )

    def _set_of_selection(self, node: SetOf, plan: QueryPlan) -> Fragment:
        """:return: the set-of's rendered selection — a plural listing for an ordered report, else
        natural Oxford-comma prose."""
        if plan.report is not None:
            return self._selection_list(node._selected_variables_, number=Number.PLURAL)
        return self._selection_list(node._selected_variables_)

    def _assemble_ranked_set_of(self, node: SetOf, plan: QueryPlan) -> Fragment:
        """:return: a ranked set-of reframed onto the entity its columns describe — *"Report, for the
        top three Employees by salary, their department and name"* — so the tuple reads as a
        possessive listing instead of a code-like *"(a, b)"*. The ranking attaches to the subject
        noun (its order key named when it is an attribute of the subject), and the columns
        pronominalise to it. A set-of whose columns share no single root keeps the bracketed tuple."""
        subject = self._tuple_subject(node, plan)
        if subject is None:
            return self._query_body(
                node,
                plan,
                self._parenthesised(node._selected_variables_),
                where_items=[self._where_clause(plan)],
                find_header=self._set_of_header(plan),
            )
        ranking = replace(
            plan.ranking,
            relation=self._key_relation_to(subject, plan.ranking.order_key),
        )
        surface = ranking_surface(RankingRequest(plan=ranking))
        subject_noun = NounPhrase(
            head=RoleFragment.for_variable(
                subject._type_.__name__, subject, number=surface.number
            ),
            number=surface.number,
            definiteness=Definiteness.DEFINITE,
            pre_head=surface.pre_head,
            modifiers=surface.modifiers,
            referent_id=_subject_id(subject),
        )
        columns = self._ranked_columns(node, ranking)
        if not columns:
            return self._query_body(
                node,
                plan,
                subject_noun,
                where_items=[self._where_clause(plan)],
                find_header=self._sentence_initial(Keywords.REPORT.as_fragment()),
            )
        header = PhraseFragment(
            parts=[
                self._sentence_initial(Keywords.REPORT.as_fragment()),
                Punctuation.COMMA.as_fragment(),
                Keywords.FOR.as_fragment(),
                subject_noun,
                Punctuation.COMMA.as_fragment(),
            ]
        )
        # Each column is rendered on its own (not owner-folded) so it pronominalises to the
        # already-introduced subject ("their department and their name"), agreeing in number.
        column_list = oxford_comma(
            [self.context.child(column) for column in columns],
            Conjunctions.AND.as_fragment(),
        )
        return self._query_body(
            node,
            plan,
            column_list,
            where_items=[self._where_clause(plan)],
            find_header=header,
        )

    def _tuple_subject(self, node: SetOf, plan: QueryPlan) -> Optional[Variable]:
        """:return: the single root variable every selected column navigates from (the entity the
        ranking reframes onto), or ``None`` when the columns share no single root."""
        if plan.discourse_root is None:
            return None
        for selection in node._selected_variables_:
            _, root = walk_chain(selection)
            if isinstance(root, Variable) and root._id_ == plan.discourse_root:
                return root
        return None

    def _key_relation_to(
        self, subject: Variable, order_key: Optional[SymbolicExpression]
    ) -> RankingKeyRelation:
        """:return: how *order_key* relates to *subject* — ``SELF`` (it is the subject), ``ATTRIBUTE``
        (a chain on it, so the ranking reads *"by <attribute>"*), or ``OTHER`` (a different root, e.g.
        an aggregate — the key is then left to the visible column)."""
        if order_key is None:
            return RankingKeyRelation.SELF
        chain, root = walk_chain(order_key)
        if not isinstance(root, Variable) or root._id_ != subject._id_:
            return RankingKeyRelation.OTHER
        return RankingKeyRelation.ATTRIBUTE if chain else RankingKeyRelation.SELF

    def _ranked_columns(
        self, node: SetOf, ranking
    ) -> List[SymbolicExpression]:
        """:return: the reported columns — the selected tuple, with the order key removed when the
        ranking already names it (*"by salary"*), so it is not listed twice."""
        selected = list(node._selected_variables_)
        names_key = ranking.relation in (
            RankingKeyRelation.ATTRIBUTE,
            RankingKeyRelation.SELF,
        )
        if names_key and ranking.order_key is not None:
            selected = [s for s in selected if s._id_ != ranking.order_key._id_]
        return selected

    def _selection_list(
        self,
        variables: List[SymbolicExpression],
        number: Number = Number.SINGULAR,
    ) -> Fragment:
        """:return: the selections joined as natural prose *"a, b, and c"* (Oxford comma, no
        parentheses) — the tuple shape reads fine without the code-like brackets. A plural *number*
        lists them as populations (*"Employees"*) for an ordered report; otherwise contiguous
        attributes of one owner fold into a shared genitive (*"the department and salary of an
        Employee"*)."""
        if number is Number.PLURAL:
            selections = [self._selected(variable, number) for variable in variables]
        else:
            selections = self._folded_selections(variables)
        return oxford_comma(selections, Conjunctions.AND.as_fragment())

    def _folded_selections(self, variables: List[SymbolicExpression]) -> List[Fragment]:
        """:return: the rendered selections, with each maximal run of plain attributes sharing one
        owner folded into a single coordinated genitive (via the shared
        :func:`~…coordination.group_consecutive_by_owner` aggregation primitive), and every other
        selection rendered alone.
        """
        fragments: List[Fragment] = []
        for item in group_consecutive_by_owner(variables, self._foldable_attribute):
            if isinstance(item, OwnerGroup):
                fragments.append(
                    coordinated_genitive(
                        [attribute_fragment(terminal) for terminal in item.items],
                        self.context.child(item.owner, inline=True),
                    )
                )
            else:
                fragments.append(self._selected(item, Number.SINGULAR))
        return fragments

    def _foldable_attribute(
        self, selection: SymbolicExpression
    ) -> Optional[Tuple[SymbolicExpression, PathStep]]:
        """:return: ``(owner, terminal_step)`` when *selection* is a plain genitive attribute that
        can share an owner with siblings, else ``None`` — a relational terminal (*"the Robot to
        which …"*) does not coordinate cleanly, so it is left alone."""
        if not isinstance(selection, Attribute):
            return None
        plan = self.context.microplan.plan_for(selection, ChainPlanner)
        if not plan.parts or plan.parts[-1].is_relation:
            return None
        return selection._child_, plan.parts[-1]

    def _selected(self, variable: SymbolicExpression, number: Number) -> Fragment:
        """:return: a single selection, as a bare plural population (*"Employees"*) when *number* is
        plural and the selection is a variable, else its default referring form."""
        if number is Number.PLURAL and isinstance(variable, Variable):
            return NounPhrase(
                head=RoleFragment.for_variable(
                    variable._type_.__name__, variable, number=Number.PLURAL
                ),
                number=Number.PLURAL,
                definiteness=Definiteness.INDEFINITE,
                referent_id=_subject_id(variable),
            )
        return self.context.child(variable)

    def _parenthesised(self, variables: List[SymbolicExpression]) -> Fragment:
        """:return: the selections as a parenthesised tuple *"(a, b)"* — for a ranked set-of, whose
        *"the top three"* pre-head needs the tuple grouped."""
        tuple_phrase = PhraseFragment(
            parts=[self.context.child(variable) for variable in variables],
            separator=Separator.COMMA,
        )
        return PhraseFragment(
            parts=[
                Punctuation.OPEN_PAREN.as_fragment(),
                tuple_phrase,
                Punctuation.CLOSE_PAREN.as_fragment(),
            ]
        )

    def _assemble_aggregation_report(
        self, node: SetOf, plan: QueryPlan, report: ReportPlan
    ) -> Fragment:
        """:return: a calculation/report — *"Report <columns>"*, or *"For each <keys>, report
        <columns>"* when grouped (the grouping stated first, so it frames the whole report and the
        trailing *"grouped by"* clause is dropped as redundant)."""
        header = (
            self._for_each_header(report.group_keys)
            if report.is_grouped
            else self._sentence_initial(Keywords.REPORT.as_fragment())
        )
        return self._query_body(
            node,
            plan,
            self._selection_list(report.columns),
            where_items=[self._where_clause(plan)],
            find_header=header,
        )

    def _assemble_grouped_report(
        self, node: Query, plan: QueryPlan, report: ReportPlan
    ) -> Fragment:
        """:return: a grouped report with no aggregates — *"For each <keys>, report all <columns>"*
        (the columns listed as per-group populations), or *"Report the distinct <keys>"* when the
        selection is exactly the group key (so there is nothing left to report but the keys
        themselves). The trailing *"grouped by"* clause is dropped as redundant."""
        if not report.columns:
            return self._query_body(
                node,
                plan,
                self._distinct_keys(report.group_keys),
                where_items=[self._where_clause(plan)],
                find_header=self._sentence_initial(Keywords.REPORT.as_fragment()),
            )
        selection = PhraseFragment(
            parts=[
                GroupingPhrases.ALL.as_fragment(),
                self._selection_list(report.columns, number=Number.PLURAL),
            ]
        )
        return self._query_body(
            node,
            plan,
            selection,
            where_items=[self._where_clause(plan)],
            find_header=self._for_each_header(report.group_keys),
        )

    def _distinct_keys(self, keys: List[SymbolicExpression]) -> Fragment:
        """:return: *"the distinct <keys>"* — the group keys as a plural population listing, for a
        grouped query that reports nothing but its keys."""
        labels = oxford_comma(
            [self._group_label(key, Number.PLURAL) for key in keys],
            Conjunctions.AND.as_fragment(),
        )
        return PhraseFragment(
            parts=[
                Articles.THE.as_fragment(),
                GroupingPhrases.DISTINCT.as_fragment(),
                labels,
            ]
        )

    def _for_each_header(self, keys: List[SymbolicExpression]) -> Fragment:
        """:return: the fronted *"For each <key>, report"* frame — the grouping first (it is the row
        dimension of the result), the keys as bare singular labels, then the lowercase verb.
        """
        labels = oxford_comma(
            [self._group_label(key) for key in keys], Conjunctions.AND.as_fragment()
        )
        return PhraseFragment(
            parts=[
                Keywords.FOR_EACH.as_fragment(),
                labels,
                Punctuation.COMMA.as_fragment(),
                Keywords.REPORT.as_fragment(),
            ]
        )

    def _group_label(
        self, key: SymbolicExpression, number: Number = Number.SINGULAR
    ) -> Fragment:
        """:return: a group key as a bare label in *number* — *"department"* / *"departments"* for an
        attribute key, the type name for a variable key — naming the group itself rather than one
        member's navigation (*"the department of an Employee"*)."""
        if isinstance(key, Attribute):
            return RoleFragment.for_attribute(
                key._owner_class_, key._attribute_name_, number=number
            )
        return RoleFragment.for_type(key._type_, number=number)

    @staticmethod
    def _sentence_initial(fragment: RoleFragment) -> Fragment:
        """:return: *fragment* with its first letter capitalised — a keyword carries its mid-sentence
        (lowercase) form, capitalised here when it opens the sentence (*"report"* → *"Report"*).
        """
        return replace(fragment, text=fragment.text[:1].upper() + fragment.text[1:])

    def _set_of_header(self, plan: QueryPlan) -> Fragment:
        """:return: The set-of header — *"Find the <top three>"* when a ``limit`` ranks the tuples
        (the order key is suppressed; it is a visible tuple element), else the plain verb
        (*"Find"* / *"Report"*). The literal *"sets of"* is intentionally dropped — the selection
        carries the set shape.
        """
        if plan.ranking is None:
            return self._verb(plan)
        surface = ranking_surface(RankingRequest(plan=plan.ranking))
        return PhraseFragment(
            parts=[
                Keywords.FIND.as_fragment(),
                Articles.THE.as_fragment(),
                surface.pre_head,
            ]
        )

    def _verb(self, plan: QueryPlan) -> Fragment:
        """:return: the opening verb — *"Report"* when the query presents results (a report),
        else *"Find"* (a search)."""
        if plan.report is not None:
            return self._sentence_initial(Keywords.REPORT.as_fragment())
        return Keywords.FIND.as_fragment()

    def _subject_number(self, plan: QueryPlan) -> Number:
        """:return: the grammatical number of the rendered subject — plural for a ranking of several
        (*"the top three Employees"*) or an ordered report (*"Report Employees"*), else singular. The
        subject's restriction and possessives agree with it (*"whose salaries are …"*).
        """
        if plan.ranking is not None:
            return ranking_surface(RankingRequest(plan=plan.ranking)).number
        if plan.report is not None and plan.report.kind in (
            ReportKind.ORDERING,
            ReportKind.GROUPING,
        ):
            return Number.PLURAL
        return Number.SINGULAR

    # ── subject selection ──────────────────────────────────────────────────────

    def _assemble_subject(self, node: Query, plan: QueryPlan) -> Fragment:
        """:return: *"Find a Robot whose battery is high, such that … [clauses]"* — the
        plain-variable selection with its WHERE woven in."""
        variable = node.selected_variable
        selected = self._build_selection(node, variable, plan)
        selected, where_items = self._apply_subject_restrictions(plan, selected)
        # No scope marker: the engine stamps this body with its query node, and the coreference
        # pass reads the focus for that query from the discourse view.
        return self._query_body(node, plan, selected, where_items=where_items)

    def _build_selection(
        self, node: Query, variable: SymbolicExpression, plan: QueryPlan
    ) -> Fragment:
        """:return: the selection's referring noun phrase — a ``limit`` ranking phrase (*"the top
        three Robots"*), else *"the unique Robot"* (``eql.the``) / *"a Robot"*."""
        if plan.ranking is not None:
            return self._build_ranking_selection(variable, plan)
        if plan.report is not None and plan.report.kind is ReportKind.ORDERING:
            # An ordered listing presents all the matching results, so the subject is plural
            # ("Report Employees …"); a plural subject pronominalises to "their".
            return self._selected(variable, self._subject_number(plan))
        if plan.is_the:
            # "the unique <type>" first mention; the coreference pass reduces a repeat to
            # "the <type>" (UNIQUE downgrades to DEFINITE) — so it is a referring noun phrase.
            return NounPhrase(
                head=RoleFragment.for_variable(plan.selected_type, variable),
                definiteness=Definiteness.UNIQUE,
                referent_id=_subject_id(variable),
            )
        # context.child(variable) → VariableRule referring noun phrase; the entity shares its referent.
        return self.context.child(variable)

    def _build_ranking_selection(
        self, variable: SymbolicExpression, plan: QueryPlan
    ) -> Fragment:
        """:return: The ranking selection — *"the first two Robots"* / *"the top three Employees by
        salary"* / *"the Employee with the highest salary"*. A ranking is inherently definite
        (*"the"*), so it ignores ``is_the``; it stays a referring noun phrase so a repeat mention
        reduces to *"the Robot"* and a WHERE pronominalises (*"its"* / *"their"*)."""
        surface = ranking_surface(RankingRequest(plan=plan.ranking))
        return NounPhrase(
            head=RoleFragment.for_variable(
                plan.selected_type, variable, number=surface.number
            ),
            number=surface.number,
            definiteness=Definiteness.DEFINITE,
            pre_head=surface.pre_head,
            modifiers=surface.modifiers,
            referent_id=_subject_id(variable),
        )

    def _apply_subject_restrictions(
        self, plan: QueryPlan, selected: Fragment
    ) -> Tuple[Fragment, List[Optional[Fragment]]]:
        """:return: The selection with its inline superlative modifiers attached, and the WHERE's
        clause items — the *"whose"* group (a sub-list of points in hierarchical) then a separate
        *"such that <residual>"* clause (each ``None`` when absent)."""
        rendered = ClauseComposer(self.context).restriction(
            plan, self._subject_number(plan)
        )
        if rendered is None:
            return selected, []
        if rendered.inline_modifiers:
            selected = PhraseFragment(parts=[selected, *rendered.inline_modifiers])
        residual = (
            PhraseFragment(parts=[Keywords.SUCH_THAT.as_fragment(), rendered.residual])
            if rendered.residual is not None
            else None
        )
        return selected, [rendered.whose, residual]

    # ── noun forms ───────────────────────────────────────────────────────────

    def _as_noun(self, entity: Query) -> Fragment:
        """
        A referring noun phrase — *"a/the unique <type>"* with the restrictions as appositive
        modifiers.

        :param entity: The nested entity selector.
        :return: The standalone-noun form *"a Robot where …"*.
        """
        plan = self.plan(entity)
        variable = entity.selected_variable
        definiteness = Definiteness.UNIQUE if plan.is_the else Definiteness.INDEFINITE

        modifiers: List[Fragment] = []
        rendered = ClauseComposer(self.context).restriction(plan)
        if rendered is not None:
            modifiers.extend(rendered.inline_modifiers)
            # A nested noun is an inline phrase, so the "whose" block flattens to "whose a, and b"
            # (the renderer expands a block into points only at the item level, not inside a phrase).
            if rendered.whose is not None:
                modifiers.append(rendered.whose)
            if rendered.residual is not None:
                modifiers.append(
                    PhraseFragment(
                        parts=[Keywords.WHERE.as_fragment(), rendered.residual]
                    )
                )

        # A referring noun phrase is the subject of its own modifiers (the coreference pass infers
        # this from the modifiers slot), so no scope marker is emitted here.
        return NounPhrase(
            head=RoleFragment.for_variable(plan.selected_type, variable),
            definiteness=definiteness,
            referent_id=_subject_id(variable),
            modifiers=modifiers,
        )

    # ── query-body clauses ─────────────────────────────────────────────────────

    def _query_body(
        self,
        node: Query,
        plan: QueryPlan,
        selection: Fragment,
        where_items: List[Optional[Fragment]],
        find_header: Optional[Fragment] = None,
    ) -> Fragment:
        """:return: *"Find <selection>"* + the present clauses (the subject restriction's *"whose"*
        / *"such that"*, then *grouped by … having … ordered by …*) as block items — absent
        clauses (``None``) are simply skipped.
        """
        if find_header is None:
            find_header = self._verb(plan)
        header = PhraseFragment(parts=[find_header, selection])
        clauses = [
            clause
            for clause in [*where_items, *self._trailing_clauses(node, plan)]
            if clause is not None
        ]
        return BlockFragment(header=header, items=clauses)

    def _trailing_clauses(
        self, node: Query, plan: QueryPlan
    ) -> List[Optional[Fragment]]:
        """:return: The post-selection clauses, in canonical reading order (``None`` when absent).

        The standalone *"ordered by …"* clause is suppressed when a ranking selection already
        conveys the ordering — i.e. a ``limit`` on a plain-variable (SUBJECT) selection. Ordering
        without a ``limit`` keeps the clause; a limited set-of keeps it too (no ranking selection).
        """
        composer = ClauseComposer(self.context)
        ranked = plan.ranking is not None and plan.kind in (
            SelectionKind.SUBJECT,
            SelectionKind.SET_OF,
        )
        # A grouped report fronts its grouping as "For each …", so the trailing "grouped by" clause
        # would merely repeat it.
        fronts_grouping = plan.report is not None and plan.report.is_grouped
        return [
            None if fronts_grouping else composer.grouped_by(node),
            composer.having(node),
            None if ranked else composer.ordered_by(node),
        ]

    def _where_clause(self, plan: QueryPlan) -> Optional[Fragment]:
        """:return: *"such that <condition>"*, or ``None`` when the query has no WHERE."""
        if plan.where_condition is None:
            return None
        return PhraseFragment(
            parts=[
                Keywords.SUCH_THAT.as_fragment(),
                self.context.child(plan.where_condition),
            ]
        )

    def inline_noun(self, entity: Query) -> Fragment:
        """
        The entity's WHERE condition is deferred to the binding scope so it can be emitted as a
        *"such that …"* clause after all binding overrides are registered.

        :param entity: The entity used as a chain root inside an instantiated variable.
        :return: The inline-noun form for *entity*.
        """
        entity.build()
        variable = entity.selected_variable
        type_name = FallbackNouns.ENTITY.name_of(variable)

        where_expression = entity._where_expression_
        if where_expression is not None:
            self.context.scope.defer_constraint(where_expression.condition)

        # A referring noun phrase (referent_id below) — a repeat reduces to "the <type>" in the pass.
        return NounPhrase(
            head=RoleFragment.for_variable(type_name, variable),
            referent_id=_subject_id(variable),
        )
