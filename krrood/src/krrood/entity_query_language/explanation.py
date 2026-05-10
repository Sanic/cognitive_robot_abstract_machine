import inspect
import uuid
import weakref
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, List, Optional, Type, Callable

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import (
        OperationResult,
        Filter,
    )


def filter_stack(
    stack: List[inspect.FrameInfo], internal_package: Optional[str] = None
) -> List[inspect.FrameInfo]:
    """
    Filter the stack to remove external libraries and optionally keep only a specific package.

    :param stack: The stack to filter.
    :param internal_package: The name of the package to focus on.
    :return: The filtered stack.
    """
    filtered = []
    for frame in stack:
        path = frame.filename
        # Exclude standard library/external packages
        if "site-packages" in path or "dist-packages" in path:
            continue

        # If a specific package is requested, filter further
        if internal_package and internal_package not in path:
            continue

        filtered.append(frame)
    return filtered


def monitored(cls: Type) -> Type:
    """
    Class decorator to automatically record the creation stack for EQL objects.

    :param cls: The class to monitor.
    :return: The monitored class.
    """
    # Inject the marker to indicate the class is monitored
    cls._is_monitored_ = True

    original_post_init = getattr(cls, "__post_init__", lambda self: None)

    @wraps(original_post_init)
    def new_post_init(self, *args, **kwargs):
        # Capture stack, skip the first few frames (decorator internal)
        raw_stack = inspect.stack()[1:]
        # Store the filtered stack on the instance
        self._creation_stack = filter_stack(raw_stack)
        original_post_init(self, *args, **kwargs)

    cls.__post_init__ = new_post_init
    return cls


@dataclass
class InferenceExplanation:
    """
    Explanation of how an instance was created through inference.
    """

    instance: Any
    """
    The instance that was created.
    """
    query_node: Any
    """
    The query node that was used to create the instance.
    """
    stack: List[inspect.FrameInfo]
    """
    The stack trace at the point of creation.
    """
    query_root: Optional[Any] = None
    """
    The root of the query that was used to create the instance.
    """
    satisfied_condition_ids: Optional[frozenset] = None
    """
    A frozenset of UUIDs of condition expressions that were satisfied (truth value = True)
    during the evaluation that produced this instance. None if no condition information is available.
    """

    def condition_graph(self):
        """
        Build a rustworkx PyDAG of the condition expression tree.

        Each node in the graph has attributes:
          - ``name``: the expression name (str)
          - ``is_satisfied``: True if the condition was satisfied, False otherwise
          - ``expression``: the SymbolicExpression object

        Edges go from child to parent, preserving the original tree structure.

        :return: A ``rustworkx.PyDAG``, or None if no conditions exist or no satisfaction
            data is available.
        """
        if not self.satisfied_condition_ids or self.query_root is None:
            return None

        condition_root = self.query_node._conditions_root_

        # If condition_root is the overall root, there are no Filter conditions
        from krrood.entity_query_language.core.base_expressions import Filter

        if not any(
            isinstance(e, Filter) for e in self.query_root._all_expressions_
        ):
            return None

        import rustworkx as rx

        graph = rx.PyDAG()
        expression_to_index: dict = {}

        def add_node(expr, parent_index=None):
            if expr._id_ in expression_to_index:
                return expression_to_index[expr._id_]

            is_satisfied = expr._id_ in self.satisfied_condition_ids
            node_index = graph.add_node(
                {
                    "name": expr._name_,
                    "is_satisfied": is_satisfied,
                    "expression": expr,
                }
            )
            expression_to_index[expr._id_] = node_index

            if parent_index is not None:
                graph.add_edge(node_index, parent_index, None)

            for child in expr._children_:
                add_node(child, node_index)

            return node_index

        add_node(condition_root)
        return graph


# Dictionary to store inference explanations for instances.
# Uses weak references to allow instances to be garbage collected.
INFERENCE_RECORD: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def register_inference(
    instance: Any, variable_node: Any, result: Optional[Any] = None
) -> None:
    """
    Register an instance created via inference into the internal records.

    :param instance: The instance to record.
    :param variable_node: The variable node that produced the instance.
    :param result: The OperationResult from the evaluation, carrying satisfied condition IDs.
    """
    # Robust check: Verify monitoring at the class level using type()
    if not getattr(type(variable_node), "_is_monitored_", False):
        return

    satisfied_ids = (
        result.satisfied_condition_ids
        if result is not None and hasattr(result, "satisfied_condition_ids")
        else None
    )
    explanation = InferenceExplanation(
        instance=instance,
        query_node=variable_node,
        # Monitored instances are guaranteed to have _creation_stack via __post_init__
        stack=variable_node._creation_stack,
        # _root_ is guaranteed by the SymbolicExpression base class
        query_root=variable_node._root_,
        satisfied_condition_ids=satisfied_ids,
    )
    try:
        INFERENCE_RECORD[instance] = explanation
    except TypeError:
        pass


def record_inferences(func: Callable) -> Callable:
    """
    Decorator for methods yielding OperationResults to record inferred instances.

    :param func: The method to decorate.
    :return: The wrapped method.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for result in func(self, *args, **kwargs):
            # If the current variable produced a binding for itself, record it
            # self._id_ is always present on SymbolicExpression
            if self._id_ in result.bindings:
                register_inference(result.bindings[self._id_], self, result)
            yield result

    return wrapper


def explain_inference(instance: Any) -> Optional[InferenceExplanation]:
    """
    Retrieve the explanation of how the given instance was created through inference.

    :param instance: The instance to explain.
    :return: An InferenceExplanation object if found, otherwise None.
    """
    try:
        return INFERENCE_RECORD.get(instance)
    except TypeError:
        return None


def format_inference_explanation(
    explanation: InferenceExplanation, focus_package: Optional[str] = None
) -> str:
    """
    Convert an InferenceExplanation into a human-readable string.

    :param explanation: The explanation object to format.
    :param focus_package: Optional package name to filter the stack further.
    :return: A formatted string explaining the inference.
    """
    # Allow further filtering at explanation time
    display_stack = filter_stack(explanation.stack, internal_package=focus_package)

    formatted_stack = []
    for frame_info in display_stack:
        formatted_stack.append(
            f'  File "{frame_info.filename}", line {frame_info.lineno}, in {frame_info.function}\n'
            f'    {frame_info.code_context[0].strip() if frame_info.code_context else "???"}\n'
        )

    stack_str = "".join(formatted_stack[:10])  # Limit to 10 frames

    return (
        f"Instance {explanation.instance} was created by inference variable: {explanation.query_node}\n"
        f"Part of query: {explanation.query_root}\n"
        f"Call stack at definition:\n{stack_str}"
    )
