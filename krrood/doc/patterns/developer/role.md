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
4. **Transparent attribute delegation.** A consumer of a role that does not know which attributes
   belong to the role and which belong to the taker should be able to read and write all of them
   through the role instance without special handling.

These goals shape every significant decision in the implementation.

## Architecture Overview

The implementation centres on three collaborating components:

**`Role[T]`** (`role.py`) is the base dataclass that every role class inherits. It provides
identity-based equality/hashing (`__hash__`, `__eq__`), attribute delegation (`__getattr__`,
`__setattr__`), and all querying class methods. It inherits from `SubClassSafeGeneric[T]` so that the generic
type parameter `T` is resolved to the concrete taker type on each concrete role subclass, and
from `Symbol` to participate in the Symbol Graph and do reasoning over it (see last point below).

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

Removing taker inheritance also eliminates the entire category of problems that arose when the
role taker class had non-trivial construction logic: the role no longer needs to replicate,
forward, or suppress that logic.

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
constructor (or to `from_role_taker`). The `__post_init__` hook never creates further roles
automatically. This makes construction auditable: you can find every role creation by searching
for calls to the role class's constructor or to `from_role_taker`.

### Field-Tag Discovery of the Role Taker

The role taker field is identified at runtime by inspecting field metadata for
`ROLE_TAKER_METADATA_KEY`. The name of that field is cached per class via `@lru_cache` on
`role_taker_field_name()`.

An alternative would have been to require a fixed field name (for example, always name the
taker field `taker`). The tag-based approach was chosen because it allows role subclasses to
choose semantically meaningful field names (`person`, `ceo`, `representative`) that convey the
domain relationship, which aids readability in both code and generated documentation.

### `SubClassSafeGeneric[T]` for Type-Level Resolution

`Role[T]` inherits from `SubClassSafeGeneric[T]`. When a concrete role class binds `T` to a
specific type — for example `CEO(Role[Person])` — `SubClassSafeGeneric` propagates that binding
through the class's field annotations before the class is fully constructed. This means
`CEO.get_role_taker_type()` returns `Person` without any runtime inspection of field values.

The alternative of inspecting `__orig_bases__` at query time would be slower and less reliable
in the presence of complex generic hierarchies (multiple type variables, TypeVarTuples, and
chains of partially-bound generics). `SubClassSafeGeneric` resolves all of these at class
definition time and caches the result.

### Distinct Identity and the `IsSameEntity` Predicate

A role is an ordinary object: `__eq__` returns `self is other` and `__hash__` is
`object.__hash__`, so each role is equal only to itself and distinct from its taker and from
sibling roles. This deliberately separates two concerns that the previous design conflated:
*object identity* (am I this exact object?) and *semantic equivalence* (do we refer to the same
underlying entity?). The latter is expressed explicitly by the `IsSameEntity` predicate in
`krrood/src/krrood/patterns/role_predicates.py`, which unwraps each operand to its
`root_persistent_entity` (walking the taker chain to the non-`Role` object) and compares the
roots by identity. Two roles at different levels of a chain, and a role versus its root taker,
are therefore *not* `==` but *are* `IsSameEntity`. A direct consequence is that multiple roles,
even of the same type, on one taker stay distinct in sets and dicts.

`__eq__` returns a concrete `False` rather than `object.__eq__`'s `NotImplemented`: because a
role delegates attribute reads to its taker, a taker with a lenient (e.g. name-based) `__eq__`
would otherwise compare equal to its role through the reflected operand. `__hash__` is set
explicitly because `Role`'s base `SubClassSafeGeneric` is a plain `@dataclass`, which would
otherwise set `__hash__ = None` (and defining `__eq__` alone would reset it to `None` too).

The predicate lives in its own module rather than in core EQL (`predicate.py`) or in `role.py`
so that neither gains a cross-dependency — `predicate.py` never imports `patterns.role` and
`role.py` never imports EQL. This mirrors how `semantic_digital_twin`'s `reasoning/predicates.py`
defines the domain predicate `ContainsType(Predicate)` in its own module.

### SymbolGraph as the Role Registry

Role lookup (checking whether a taker has a role, retrieving roles of a given type) is
implemented as a graph traversal over the `SymbolGraph`. The role registers itself by adding a
`HasRoleTaker` edge from itself to its taker during `__post_init__`.

When the taker is itself a role, `_update_mapping_between_roles_and_role_takers` additionally
adds transitive `HasRoleTaker` edges from the new role to every entity that the taker's taker
already points to. This ensures that querying from any point in the chain resolves to all roles
and takers in the chain.

Classes that are not registered in the class diagram (for example, dynamically created test
classes) are handled gracefully: `ClassIsUnMappedInClassDiagram` is caught and the registration
step is silently skipped, leaving the role functional as a plain dataclass without graph
integration.

## Attribute Delegation

`__getattr__` delegates attribute reads that failed normal lookup to the role taker. Because
`__getattr__` is only called when the standard attribute resolution chain has already failed,
role-native attributes (those declared as dataclass fields on the role class) are never affected.

`__setattr__` is more nuanced. During `__post_init__`, the role taker field is not yet set when
the first `super().__setattr__` call occurs. The `_role_taker_is_set` guard prevents premature
delegation during that window. Once the taker field is set, writes to attributes that exist on
the taker are delegated to the taker, and writes to attributes that exist only on the role (or
neither) are stored on the role.

## Source References

- `krrood/src/krrood/patterns/role.py` — `Role`, `role_taker_field`, `HasRoleTaker`,
  `RoleTakerFieldNotFound`
- `krrood/src/krrood/patterns/subclass_safe_generic.py` — `SubClassSafeGeneric`,
  `AbstractSubClassSafeGeneric`
- `krrood/src/krrood/class_diagrams/utils.py` — `ROLE_TAKER_METADATA_KEY`
- `krrood/src/krrood/symbol_graph/symbol_graph.py` — `SymbolGraph`, `PredicateClassRelation`,
  `Symbol`
- `test/krrood_test/test_patterns/test_role.py` — behavioural tests for the role pattern
- `test/krrood_test/dataset/role_and_ontology/university_ontology_like_classes_without_descriptors.py`
  — canonical fixture for role pattern tests
