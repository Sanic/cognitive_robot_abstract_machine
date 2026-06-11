"""
Chain **assembler** — realise a
:class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable` chain
(Attribute / Index / Call) into its surface phrase.

Chains have no separate *plan*: there is no content selection to decide, only the
surface form to render (a boolean terminal attribute → predicative *"<nav> is [not]
<attr>"*; anything else → the possessive path *"the attr of the Root"*, optionally
pronominalised to *"its …"* when the root is the current coreference subject).  It is
therefore a realisation-only :class:`Assembler` (``Assembler[None]``); its sub-steps are
methods sharing ``self.ctx`` (recursion via ``self.ctx.child``).  Entity-rooted chains
defer to :meth:`QueryAssembler.inline_noun`.

Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
"""

from __future__ import annotations

from typing_extensions import Optional

from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Index,
    MappedVariable,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.chain_utils import (
    build_path_parts,
    ChainAnalysis,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    PossessiveChain,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.assembly.query import (
    QueryAssembler,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    possessive_path,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Copulas,
    Prepositions,
)


class ChainAssembler(Assembler[MappedVariable, None]):
    """Realise a MappedVariable chain (possessive / predicative / pronominal forms).

    Realisation-only (``planner = None``): a chain has no content to decide, only a surface
    form, so there is no plan — :meth:`realize` ignores it.
    """

    def realize(self, node, plan: None = None) -> VerbFragment:
        """Default chain rendering (the :class:`Assembler` entry point)."""
        return self.chain(node)

    # ── entry forms ──────────────────────────────────────────────────────────

    def chain(
        self, expression: MappedVariable, *, negated: bool = False
    ) -> VerbFragment:
        """Boolean terminal → predicative *"<nav> is [not] <attr>"*; else possessive path.

        When a plural is requested (``ctx.number``) and this is a single attribute on a
        variable, build the bare plural *"attrs of Roots"*; otherwise render singular.  The
        chain is analysed once (:class:`ChainAnalysis`) and each branch reads off that value.
        """
        analysis = ChainAnalysis.of(expression)
        if self.ctx.number is Number.PLURAL:
            plural = self._plural_attribute(analysis)
            if plural is not None:
                return plural
        if analysis.is_bool_terminal:
            return self._bool_predicative(analysis, negated)
        root_fragment = self._chain_root(analysis.root)
        if isinstance(analysis.root, Variable):
            # Defer the pronominal-vs-possessive choice to the coreference pass: it knows
            # whether the root is the current subject (a build-time fact no longer consulted here).
            return PossessiveChain(
                parts=analysis.parts,
                root_fragment=root_fragment,
                root_referent_id=analysis.root._id_,
            )
        return possessive_path(analysis.parts, root_fragment)

    def _plural_attribute(self, analysis: ChainAnalysis) -> Optional[VerbFragment]:
        """*"attrs of Roots"* when the chain is a single ``Attribute`` on a ``Variable``,
        else ``None`` (caller falls through to the singular rendering).  Tags both leaves
        plural for the morphology pass; marks the root introduced for cross-build seeding.
        """
        root = analysis.root
        if not (
            isinstance(root, Variable)
            and len(analysis.chain) == 1
            and isinstance(analysis.chain[0], Attribute)
        ):
            return None
        label, numbered = self.ctx.refer.numbered_label(root)
        attribute = analysis.chain[0]
        root_np = NounPhrase(
            head=RoleFragment.for_variable(label, root),
            number=Number.SINGULAR if numbered else Number.PLURAL,
            definiteness=Definiteness.BARE if numbered else Definiteness.INDEFINITE,
            referent_id=root._id_,
        )
        return NounPhrase(
            head=RoleFragment.for_attribute(
                attribute._owner_class_, attribute._attribute_name_
            ),
            number=Number.PLURAL,
            definiteness=Definiteness.INDEFINITE,
            modifiers=[Prepositions.OF.as_fragment(), root_np],
        )

    def _chain_root(self, leaf: object) -> VerbFragment:
        """Noun phrase for the chain root — inline-noun for Entity roots, else recurse."""
        inner = leaf
        while isinstance(inner, ResultQuantifier):
            inner = inner._child_
        if isinstance(inner, Entity):
            return QueryAssembler(self.ctx).inline_noun(inner)
        return self.ctx.child(leaf)

    def _bool_predicative(self, analysis: ChainAnalysis, negated: bool) -> VerbFragment:
        """*"<nav> is [not] <attr>"* — chains ending in an int Index get ordinal navigation."""
        chain = analysis.chain
        root_fragment = self._chain_root(analysis.root)
        nav_chain = chain[:-1]

        if not nav_chain:
            nav_fragment = root_fragment
        elif isinstance(nav_chain[-1], Index) and isinstance(nav_chain[-1]._key_, int):
            ordinal = morphology.ordinal(nav_chain[-1]._key_)
            pre_frag = possessive_path(build_path_parts(nav_chain[:-1]), root_fragment)
            nav_fragment = PhraseFragment(
                parts=[
                    Articles.THE.as_fragment(),
                    WordFragment(text=ordinal),
                    Prepositions.OF.as_fragment(),
                    pre_frag,
                ]
            )
        else:
            nav_fragment = possessive_path(build_path_parts(nav_chain), root_fragment)

        copula = Copulas.IS_NOT.as_fragment() if negated else Copulas.IS.as_fragment()
        terminal = chain[-1]
        return PhraseFragment(
            parts=[
                nav_fragment,
                copula,
                RoleFragment.for_attribute(
                    terminal._owner_class_, terminal._attribute_name_
                ),
            ]
        )
