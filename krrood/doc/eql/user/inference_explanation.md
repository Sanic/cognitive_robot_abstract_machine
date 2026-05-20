---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Inference Explanation

When EQL infers a new object via an `inference(...)` rule, it automatically attaches an
**`InferenceExplanation`** to that object. The explanation records:

- **Which query node** produced the instance.
- **Which conditions** in the query were satisfied (and their truth-value bindings).
- **The full call stack** at the point where the query was written.
- The **`OperationResult`** from the evaluation, carrying the complete variable bindings.

You retrieve it with `explain_inference`:

```python
from krrood.entity_query_language.explanation.explanation import explain_inference

drawers = query.tolist()
explanation = explain_inference(drawers[0])
```

`explain_inference` returns `None` for instances that were not produced by an inference variable
(e.g. plain instances constructed directly).

---

## Simple Usage

### Human-readable summary

```python
print(explanation.as_string())
# Instance ExampleDrawer(...) was created by inference variable: ...
# Part of query: ...
# Call stack at definition:
#   File "my_script.py", line 12, in <module>
#     drawers = query.tolist()
```

Pass `focus_package` to filter the call stack to your own code:

```python
print(explanation.as_string(focus_package="my_package"))
```

### Satisfied conditions

```python
print(explanation.get_satisfied_conditions_as_string())
# (fixed_conn.parent == body)
# AND (fixed_conn.child == handle)
```

### Condition graph

```python
graph = explanation.condition_graph()   # QueryGraph or None
```

The graph carries `is_satisfied` flags on every condition node, ready for visualization.
See {doc}`graph_and_visualization <../developer/graph_and_visualization>` for rendering details.

---

## Meta-queries

`InferenceExplanation` inherits from `Symbol`, making it a first-class entity in the
`SymbolGraph`. Its methods return EQL **Entity** descriptors that can be chained, filtered,
and composed like ordinary queries.

### Which conditions were satisfied?

```python
conditions = explanation.get_satisfied_condition_expressions_for_the_instance().tolist()
# → list of Comparator / InstantiatedVariable expressions
```

### Which variable nodes participated (by type)?

```python
conn_nodes = explanation.get_variable_nodes_of_given_type(ExampleFixedConnection).tolist()
# → [Variable(_type_=ExampleFixedConnection)]
```

### What were the actual bound values?

```python
handles = explanation.get_values_of_variable_nodes_of_given_type(ExampleHandle).tolist()
# → [ExampleHandle(name='H1')]
```

### Which conditions relate two variables of the same type?

```python
# Conditions whose descendant tree contains ≥2 distinct ExampleBody-typed variable nodes
body_conditions = explanation.get_conditions_that_relate_the_variables_of_type(ExampleBody).tolist()
```

### Which conditions relate variables of two different types?

```python
# Conditions that mention both an ExampleFixedConnection and an ExampleHandle variable
cross_conditions = explanation.get_conditions_that_relate_variables_of_types(
    ExampleFixedConnection, ExampleHandle
).tolist()
```

`get_conditions_that_relate_variables_of_types` is symmetric: swapping the two types returns
the same set of conditions.

---

## Full Example

```{code-cell} ipython3
from dataclasses import dataclass
from krrood.entity_query_language.factories import (
    variable, entity, Symbol, deduced_variable, add, inference,
)
from krrood.entity_query_language.explanation.explanation import explain_inference


@dataclass
class ExampleHandle(Symbol):
    name: str


@dataclass
class ExampleBody(Symbol):
    name: str


@dataclass
class ExampleFixedConnection(Symbol):
    parent: ExampleBody
    child: ExampleHandle


@dataclass
class ExampleDrawer(Symbol):
    body: ExampleBody
    handle: ExampleHandle


# --- data ---
bodies = [ExampleBody("big_body"), ExampleBody("small_body")]
handles = [ExampleHandle("H1"), ExampleHandle("H2")]
connections = [
    ExampleFixedConnection(parent=bodies[0], child=handles[0]),
    ExampleFixedConnection(parent=bodies[1], child=handles[1]),
]

# --- query ---
fixed = variable(ExampleFixedConnection, domain=connections)
drawer = deduced_variable(ExampleDrawer)

query = entity(drawer).where(fixed.parent.name.startswith("big"))

with query:
    add(drawer, inference(ExampleDrawer)(body=fixed.parent, handle=fixed.child))

# --- execute ---
results = query.tolist()
print(f"Inferred {len(results)} drawer(s).")

# --- explain ---
expl = explain_inference(results[0])
print("\n--- as_string() ---")
print(expl.as_string())

print("\n--- get_satisfied_conditions_as_string() ---")
print(expl.get_satisfied_conditions_as_string())

print("\n--- variable nodes of type ExampleFixedConnection ---")
for node in expl.get_variable_nodes_of_given_type(ExampleFixedConnection).tolist():
    print(" ", node)

print("\n--- bound values of type ExampleHandle ---")
for val in expl.get_values_of_variable_nodes_of_given_type(ExampleHandle).tolist():
    print(" ", val)
```

## API Reference
- {py:func}`~krrood.entity_query_language.explanation.explanation.explain_inference`
- {py:class}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.as_string`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_satisfied_conditions_as_string`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.condition_graph`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_satisfied_condition_expressions_for_the_instance`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_variable_nodes_of_given_type`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_values_of_variable_nodes_of_given_type`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_conditions_that_relate_the_variables_of_type`
- {py:meth}`~krrood.entity_query_language.explanation.explanation.InferenceExplanation.get_conditions_that_relate_variables_of_types`
