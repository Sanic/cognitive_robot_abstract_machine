"""
Rule registry — populated automatically by
:class:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule.__init_subclass__`.

Every concrete :class:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule`
subclass self-registers when its module is imported.  Import all rule modules below to
trigger registration, then export :data:`ALL_RULES` as the sorted list consumed by
:class:`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer`.

To add a new rule, create a class in a ``rules/*.py`` file (or a new file imported here).
No manual list maintenance is needed.
"""

from __future__ import annotations

from typing import List
from typing_extensions import Type

from krrood.entity_query_language.verbalization.rule_engine import VerbalizationRule

ALL_RULES: List[Type[VerbalizationRule]] = VerbalizationRule.registered_rules()
