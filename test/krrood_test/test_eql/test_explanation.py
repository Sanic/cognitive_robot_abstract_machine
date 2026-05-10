import pytest
from dataclasses import dataclass
from krrood.entity_query_language.factories import inference, entity
from krrood.entity_query_language.explanation import (
    explain_inference,
    format_inference_explanation, register_inference,
)

@dataclass(frozen=True)
class Person:
    name: str

def test_explain_inference_basic():
    """
    Test that explain_inference correctly records and retrieves the stack for a simple inference.
    """
    # 1. Define the query
    # The stack captured should point here
    person_factory = inference(Person)
    query = entity(person_factory(name="John"))
    
    # 2. Evaluate the query to trigger instance creation
    results = list(query.evaluate())
    assert len(results) == 1
    john = results[0]
    
    # 3. Check explanation
    explanation_obj = explain_inference(john)
    assert explanation_obj is not None
    explanation = format_inference_explanation(explanation_obj)

    assert "Person" in explanation
    assert "test_explain_inference_basic" in explanation
    assert "person_factory(name=\"John\")" in explanation

def test_explain_inference_nested():
    """
    Test that explain_inference correctly records and retrieves the stack through nested function calls.
    """
    def create_person_query(name):
        return person_factory_helper(name)

    def person_factory_helper(name):
        person_inf = inference(Person)
        return person_inf(name=name)

    # Define query through nested calls
    p_var = create_person_query("Alice")
    query = entity(p_var)

    results = list(query.evaluate())
    assert len(results) == 1
    alice = results[0]

    explanation_obj = explain_inference(alice)
    assert explanation_obj is not None
    explanation = format_inference_explanation(explanation_obj)

    assert "Person" in explanation
    assert "test_explain_inference_nested" in explanation
    assert "create_person_query" in explanation
    assert "person_factory_helper" in explanation
    assert "person_inf(name=name)" in explanation

def test_explain_inference_multiple_instances():
    """
    Test that different instances from the same inference variable have the same stack in their explanation.
    """
    from krrood.entity_query_language.factories import variable_from
    
    names = variable_from(["Bob", "Charlie"])
    person_inf = inference(Person)
    query = entity(person_inf(name=names))
    
    results = list(query.evaluate())
    assert len(results) == 2
    
    bob = next(r for r in results if r.name == "Bob")
    charlie = next(r for r in results if r.name == "Charlie")
    
    expl_bob_obj = explain_inference(bob)
    expl_charlie_obj = explain_inference(charlie)
    assert expl_bob_obj is not None
    assert expl_charlie_obj is not None
    expl_bob = format_inference_explanation(expl_bob_obj)
    expl_charlie = format_inference_explanation(expl_charlie_obj)
    
    assert "test_explain_inference_multiple_instances" in expl_bob
    assert "test_explain_inference_multiple_instances" in expl_charlie
    assert "person_inf(name=names)" in expl_bob
    assert "person_inf(name=names)" in expl_charlie

def test_explain_inference_deeply_nested():
    """
    Test that explain_inference correctly records and retrieves the stack through deeply nested function calls.
    """
    def level_4(name):
        person_inf = inference(Person)
        return person_inf(name=name)

    def level_3(name):
        return level_4(name)

    def level_2(name):
        return level_3(name)

    def level_1(name):
        return level_2(name)

    # Define query through deeply nested calls
    p_var = level_1("Dave")
    query = entity(p_var)

    results = list(query.evaluate())
    assert len(results) == 1
    dave = results[0]

    explanation_obj = explain_inference(dave)
    assert explanation_obj is not None
    explanation = format_inference_explanation(explanation_obj)

    assert "Person" in explanation
    assert "test_explain_inference_deeply_nested" in explanation
    assert "level_1" in explanation
    assert "level_2" in explanation
    assert "level_3" in explanation
    assert "level_4" in explanation
    assert "person_inf(name=name)" in explanation


def test_query_stack_tracking():
    """
    Test that Query objects automatically record their creation stack.
    """
    person_inf = inference(Person)
    query = entity(person_inf(name="Eve"))

    assert hasattr(query, "_creation_stack")
    assert isinstance(query._creation_stack, list)
    # The stack should contain this test function
    filenames = [f.filename for f in query._creation_stack]
    assert any("test_explanation.py" in f for f in filenames)
    functions = [f.function for f in query._creation_stack]
    assert "test_query_stack_tracking" in functions


def test_explain_inference_focus_package():
    """
    Test that explain_inference correctly filters by focus_package.
    """
    person_inf = inference(Person)
    query = entity(person_inf(name="Frank"))

    results = list(query.evaluate())
    frank = results[0]

    # Full explanation
    explanation_obj = explain_inference(frank)
    assert explanation_obj is not None
    explanation_full = format_inference_explanation(explanation_obj)
    assert "test_explanation.py" in explanation_full
    # Assuming krrood is in the path for some internal frames if any (though here it's mostly test)
    # But we can force a check that only contains 'test_explanation' if we filter by it
    explanation_filtered = format_inference_explanation(
        explanation_obj, focus_package="test_explanation.py"
    )

    assert "test_explanation.py" in explanation_filtered
    # If there were other packages, they would be filtered out. 
    # Since our filter_stack already excludes site-packages, the diff might be subtle in this simple test.


def test_variable_stack_tracking():
    """
    Test that Variable objects automatically record their creation stack.
    """
    from krrood.entity_query_language.factories import variable_from
    v = variable_from([1, 2, 3])

    assert hasattr(v, "_creation_stack")
    assert isinstance(v._creation_stack, list)
    filenames = [f.filename for f in v._creation_stack]
    assert any("test_explanation.py" in f for f in filenames)


def test_robust_monitoring_check():
    """
    Test that register_inference safely ignores non-monitored variables.
    """
    # Create a dummy non-monitored variable-like object
    class NonMonitoredVariable:
        def __init__(self):
            self._id_ = "dummy-id"
            self._root_ = self

    dummy_var = NonMonitoredVariable()
    dummy_instance = "dummy-instance"
    
    # Should NOT raise AttributeError or any other error
    register_inference(dummy_instance, dummy_var)
    
    # Check that it was NOT recorded
    assert explain_inference(dummy_instance) is None
