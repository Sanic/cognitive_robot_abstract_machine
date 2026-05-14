"""
Runtime tests for the HasRoles mixin and roles-dict population.
Uses the ground-truth transformed classes, which inherit from HasRoles.
"""

from __future__ import annotations

import pytest

from krrood.patterns.role import HasRoles
from ..dataset.role_and_ontology._ground_truth_transformed_university_ontology_like_classes_without_descriptors import (
    PersonInRoleAndOntology,
    CEOAsFirstRole,
    ProfessorAsFirstRole,
    RepresentativeAsSecondRole,
    Company,
    Course,
)


@pytest.fixture
def person():
    return PersonInRoleAndOntology(name="Alice")


def test_role_taker_is_has_roles_instance(person):
    assert isinstance(person, HasRoles)


def test_roles_dict_empty_before_any_role_created(person):
    assert person.roles == {}


def test_roles_dict_populated_on_role_creation(person):
    ceo = CEOAsFirstRole(person=person)
    assert CEOAsFirstRole in person.roles
    assert person.roles[CEOAsFirstRole] is ceo


def test_roles_dict_contains_correct_instance(person):
    ceo = CEOAsFirstRole(person=person, head_of=Company(name="ACME"))
    assert person.roles[CEOAsFirstRole].head_of == Company(name="ACME")


def test_roles_dict_grows_with_each_role(person):
    ceo = CEOAsFirstRole(person=person)
    prof = ProfessorAsFirstRole(person=person)
    assert len(person.roles) == 2
    assert person.roles[CEOAsFirstRole] is ceo
    assert person.roles[ProfessorAsFirstRole] is prof


def test_role_not_in_dict_when_not_created(person):
    CEOAsFirstRole(person=person)
    assert ProfessorAsFirstRole not in person.roles


def test_role_chain_taker_also_gets_registered():
    """RepresentativeAsSecondRole's taker is a CEOAsFirstRole which is itself a HasRoles."""
    person = PersonInRoleAndOntology(name="Bob")
    ceo = CEOAsFirstRole(person=person)
    rep = RepresentativeAsSecondRole(ceo=ceo)
    assert RepresentativeAsSecondRole in ceo.roles
    assert ceo.roles[RepresentativeAsSecondRole] is rep


def test_role_attrs_accessible_via_roles_dict(person):
    """Role-native attributes are accessible through the roles dict."""
    ceo = CEOAsFirstRole(person=person, head_of=Company(name="ACME"))
    prof = ProfessorAsFirstRole(person=person)
    prof.teacher_of.append(Course(name="Math"))

    assert person.roles[CEOAsFirstRole].head_of == Company(name="ACME")
    assert person.roles[ProfessorAsFirstRole].teacher_of[0].name == "Math"


def test_shared_base_attr_still_delegates(person):
    """Shared-base attrs (from HasName) still delegate from role to taker."""
    ceo = CEOAsFirstRole(person=person)
    assert ceo.name == person.name
    ceo.name = "NewName"
    assert person.name == "NewName"
