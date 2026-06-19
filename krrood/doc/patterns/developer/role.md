---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Role Pattern — Developer Guide

This document describes the design goals, architectural decisions, and extension points of the
Role pattern as implemented in `krrood.patterns.role`.

## Architectural Goals and Constraints

The Role pattern solves a modelling problem that arises in knowledge representation: a single
real-world entity can participate in many different relationships and carry many different
context-specific properties, but its *identity* remains constant throughout. Subclassing cannot
express this because a subclass implies a permanent, type-level distinction. Association cannot
express it because an associated object is a separate entity with its own identity.

The design optimises for four properties:

1. **Distinct identity, explicit equivalence.** A role is an ordinary object with its own
   identity (equal only to itself), so multiple roles — even of the same type — on one taker stay
   distinct. "Same underlying entity?" is answered explicitly by the `IsSameEntity` predicate
   rather than by overloading `==`/`hash`.
2. **Explicit, auditable construction.** The system that creates a role must pass the role taker
   explicitly. There must be no implicit or automatic role creation.
3. **Pure composition, not inheritance.** A role class must not inherit from its role taker type.
   Role membership is expressed through registry queries, not `isinstance` checks.
4. **Transparent attribute reads.** A consumer of a role that does not know which attributes belong
   to the role and which belong to the taker should be able to read all of them through the role
   instance without special handling. Writes are deliberately not transparent: an assignment always
   targets the role, so changing the taker is an explicit operation through `role.role_taker`. This
   keeps writes unambiguous and prevents mutating a shared entity as a side effect of writing
   through one of its roles.

These goals shape every significant decision in the implementation.

## Architecture Overview

The implementation centres on three collaborating components:

**`Role[T]`** (`role.py`) is the base dataclass that every role class inherits. It provides
identity-based equality/hashing (`__hash__`, `__eq__`), attribute delegation (`__getattr__`,
`__setattr__`), and all querying class methods. It inherits from `SubClassSafeGeneric[T]` so that
fields whose annotation uses the generic parameter are rewritten to the bound concrete type on each
role subclass (keeping generic roles introspectable by the class diagram), and from `Symbol` to
participate in the Symbol Graph and do reasoning over it (see last point below).

**`role_taker_field()`** is a thin wrapper around `dataclasses.field()` that tags the field in
its metadata with `ROLE_TAKER_METADATA_KEY`. This tag is the single source of truth that the
role uses to identify which field is the role taker. The tag also allows the class diagram
(`krrood.class_diagrams`) to recognise role-taker relationships during graph construction.

**`SymbolGraph`** (external, in `krrood.symbol_graph`) is the runtime registry. When a role is
constructed, `Role.__post_init__` calls
`_update_mapping_between_roles_and_role_takers`, which writes a `HasRoleTaker` edge into the
symbol graph linking the wrapped role to its wrapped taker. Querying whether a taker has a
given role type then becomes a graph traversal over incoming `HasRoleTaker` edges.

## Key Design Decisions and Rationale

### Pure Composition: No Inheritance from the Taker Type

The role class does not inherit from the taker class.

The reason is that inheritance implies substitutability: if `CEO` inherited from `Person`, every
`isinstance(ceo, Person)` check would return `True`, and every API that accepted a `Person`
would silently accept a `CEO`. This causes hidden coupling between systems that should not need
to know about each other's roles. Role membership must be queried explicitly through
`Role.has_role` or `Role.roles_for`, which makes the dependency visible.

Pure composition also keeps the role independent of the taker's construction logic.

### No `__init__` Manipulation

The `Role` class uses the standard dataclass `__post_init__` hook for all post-construction work.
There is no override of `__init__`, no metaclass manipulation of `__init__`, and no
`InitVar`-based workarounds.

This means that the constructor signature of every concrete role class is exactly what
`@dataclass` generates from its fields. A caller can always read a role class's field
declarations and know exactly what arguments its constructor accepts, with no hidden
transformations.

### Explicit Construction Only

A role is only created when calling code explicitly passes a role taker instance to the role
constructor (or to `from_role_taker`).

### Field-Tag Discovery of the Role Taker

The role taker field is identified at runtime by inspecting field metadata for
`ROLE_TAKER_METADATA_KEY`. The name of that field is cached per class via `@lru_cache` on
`role_taker_field_name()`.

### Distinct Identity and the `IsSameEntity` Predicate

Each role is equal only to itself and distinct from its taker and from
sibling roles. This keeps two concerns separate: *object identity* (am I this exact object?) and
*semantic equivalence* (do we refer to the same underlying entity?). The latter is expressed explicitly by the `IsSameEntity` predicate in
`krrood/src/krrood/patterns/role_predicates.py`, which unwraps each operand to its
`root_persistent_entity` (walking the taker chain to the non-`Role` object) and compares the
roots by identity. Two roles at different levels of a chain, and a role versus its root taker,
are therefore *not* `==` but *are* `IsSameEntity`. A direct consequence is that multiple roles,
even of the same type, on one taker stay distinct in sets and dicts.

### SymbolGraph as the Role Registry

Role lookup (checking whether a taker has a role, retrieving roles of a given type) is
implemented as a graph traversal over the `SymbolGraph`. The role registers itself by adding a
`HasRoleTaker` edge from itself to its taker during `__post_init__`.

When the taker is itself a role, `_update_mapping_between_roles_and_role_takers` additionally
adds transitive `HasRoleTaker` edges from the new role to every entity that the taker's taker
already points to. This ensures that querying from any point in the chain resolves to all roles
and takers in the chain.

## Attribute Access

`__getattr__` delegates attribute reads that failed normal lookup to the role taker. Because
`__getattr__` is only called when the standard attribute resolution chain has already failed,
role-native attributes (those declared as dataclass fields on the role class) are read from the
role directly and never reach the taker.

`__setattr__` always sets attributes on the role itself, never on the role taker. Only the role's
own declared fields and private (underscore-prefixed) attributes may be assigned; any other name
raises `RoleAttributeNotDeclaredError`. Reads are delegated but writes are not, so writing through a
role cannot mutate the shared entity as a side effect, and the rejection of undeclared names keeps a
write from silently shadowing a role-taker attribute. Code that needs to modify the taker does so
explicitly through `role.role_taker` (or `role.root_persistent_entity`).

## Source References

- `krrood/src/krrood/patterns/role.py` — `Role`, `role_taker_field`, `HasRoleTaker`
- `krrood/src/krrood/patterns/exceptions.py` — `RoleTakerFieldNotFound`,
  `DelegatedFactoryMethodError`, `RoleAttributeNotDeclaredError`
- `krrood/src/krrood/patterns/subclass_safe_generic.py` — `SubClassSafeGeneric`,
  `AbstractSubClassSafeGeneric`
- `krrood/src/krrood/class_diagrams/utils.py` — `ROLE_TAKER_METADATA_KEY`
- `krrood/src/krrood/symbol_graph/symbol_graph.py` — `SymbolGraph`, `PredicateClassRelation`,
  `Symbol`
- `test/krrood_test/test_patterns/test_role.py` — behavioural tests for the role pattern
- `test/krrood_test/dataset/role_and_ontology/university_ontology_like_classes_without_descriptors.py`
  — canonical fixture for role pattern tests
