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

from typing_extensions import List, Optional, Tuple

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
    walk_chain,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import phrase, word
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.assembly.query import (
    QueryAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_bool_attr_chain,
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
        variable, build the bare plural *"attrs of Roots"*; otherwise render singular.
        """
        if self.ctx.number is Number.PLURAL:
            plural = self._plural_attribute(expression)
            if plural is not None:
                return plural
        chain, leaf = walk_chain(expression)
        if is_bool_attr_chain(expression):
            return self._bool_predicative(chain, leaf, negated)
        root_fragment = self._chain_root(leaf)
        return self._possessive_path(build_path_parts(chain), root_fragment)

    def _plural_attribute(self, expression: MappedVariable) -> Optional[VerbFragment]:
        """*"attrs of Roots"* when *expression* is a single ``Attribute`` on a ``Variable``,
        else ``None`` (caller falls through to the singular rendering).  Tags both leaves
        plural for the morphology pass; registers the root for coreference."""
        chain, root = walk_chain(expression)
        if not (
            isinstance(root, Variable)
            and len(chain) == 1
            and isinstance(chain[0], Attribute)
        ):
            return None
        type_name = root._type_.__name__
        label = self.ctx.refer.disambiguation_map.get(root._id_, type_name)
        self.ctx.refer.register_label(root, label)
        numbered = label != type_name
        attribute = chain[0]
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

    def possessive(
        self, expression: MappedVariable, pronoun: VerbFragment
    ) -> VerbFragment:
        """Chain rooted at the coreference subject → *"its booking_date"* / *"the amount of its …"*."""
        chain, _root = walk_chain(expression)
        return self._pronominal_path(build_path_parts(chain), pronoun)

    # ── path builders ──────────────────────────────────────────────────────────

    def _possessive_path(
        self, parts: List[Tuple[str, Optional[SourceRef]]], root_fragment: VerbFragment
    ) -> VerbFragment:
        """*"the <inner> of the <outer> of <root>"* (parts iterated innermost-first)."""
        if not parts:
            return root_fragment
        reversed_parts = list(reversed(parts))
        first_name, first_ref = reversed_parts[0]
        fragment_parts: List[VerbFragment] = [
            Articles.THE.as_fragment(),
            self._attr(first_name, first_ref),
        ]
        for attribute_name, attribute_reference in reversed_parts[1:]:
            fragment_parts.extend(
                [
                    Prepositions.OF_THE.as_fragment(),
                    self._attr(attribute_name, attribute_reference),
                ]
            )
        fragment_parts.extend([Prepositions.OF.as_fragment(), root_fragment])
        return PhraseFragment(parts=fragment_parts)

    def _pronominal_path(
        self, parts: List[Tuple[str, Optional[SourceRef]]], pronoun: VerbFragment
    ) -> VerbFragment:
        """*"its attr"* (single hop) or *"the attr of its foo"* (multi-hop)."""
        if not parts:
            return pronoun
        reversed_parts = list(reversed(parts))
        last = len(reversed_parts) - 1
        fragment_parts: List[VerbFragment] = []
        for index, (attribute_name, attribute_reference) in enumerate(reversed_parts):
            attribute_fragment = self._attr(attribute_name, attribute_reference)
            if index == 0 and index != last:
                fragment_parts.extend([Articles.THE.as_fragment(), attribute_fragment])
            elif index == 0:  # single attribute → "its booking_date"
                fragment_parts.extend([pronoun, attribute_fragment])
            elif index == last:  # adjacent to the elided root → "of its amount_details"
                fragment_parts.extend(
                    [Prepositions.OF.as_fragment(), pronoun, attribute_fragment]
                )
            else:
                fragment_parts.extend(
                    [Prepositions.OF_THE.as_fragment(), attribute_fragment]
                )
        return PhraseFragment(parts=fragment_parts)

    def _attr(self, name: str, source_ref: Optional[SourceRef]) -> RoleFragment:
        """A role-tagged attribute fragment."""
        return RoleFragment(
            text=name, role=SemanticRole.ATTRIBUTE, source_ref=source_ref
        )

    def _chain_root(self, leaf: object) -> VerbFragment:
        """Noun phrase for the chain root — inline-noun for Entity roots, else recurse."""
        inner = leaf
        while isinstance(inner, ResultQuantifier):
            inner = inner._child_
        if isinstance(inner, Entity):
            return QueryAssembler(self.ctx).inline_noun(inner)
        return self.ctx.child(leaf)

    def _bool_predicative(
        self, chain: List[MappedVariable], leaf: object, negated: bool
    ) -> VerbFragment:
        """*"<nav> is [not] <attr>"* — chains ending in an int Index get ordinal navigation."""
        root_fragment = self._chain_root(leaf)
        nav_chain = chain[:-1]

        if not nav_chain:
            nav_fragment = root_fragment
        elif isinstance(nav_chain[-1], Index) and isinstance(nav_chain[-1]._key_, int):
            ordinal = morphology.ordinal(nav_chain[-1]._key_)
            pre_frag = self._possessive_path(
                build_path_parts(nav_chain[:-1]), root_fragment
            )
            nav_fragment = phrase(
                Articles.THE.as_fragment(),
                word(ordinal),
                Prepositions.OF.as_fragment(),
                pre_frag,
            )
        else:
            nav_fragment = self._possessive_path(
                build_path_parts(nav_chain), root_fragment
            )

        copula = Copulas.IS_NOT.as_fragment() if negated else Copulas.IS.as_fragment()
        terminal = chain[-1]
        return phrase(
            nav_fragment,
            copula,
            RoleFragment.for_attribute(
                terminal._owner_class_, terminal._attribute_name_
            ),
        )
