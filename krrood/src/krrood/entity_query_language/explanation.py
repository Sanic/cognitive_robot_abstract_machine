import inspect
import weakref
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, List, Optional, Type, Callable


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


# Dictionary to store inference explanations for instances.
# Uses weak references to allow instances to be garbage collected.
INFERENCE_RECORD: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def register_inference(instance: Any, variable_node: Any) -> None:
    """
    Register an instance created via inference into the internal records.

    :param instance: The instance to record.
    :param variable_node: The variable node that produced the instance.
    """
    # Robust check: Verify monitoring at the class level using type()
    if not getattr(type(variable_node), "_is_monitored_", False):
        return

    explanation = InferenceExplanation(
        instance=instance,
        query_node=variable_node,
        # Monitored instances are guaranteed to have _creation_stack via __post_init__
        stack=variable_node._creation_stack,
        # _root_ is guaranteed by the SymbolicExpression base class
        query_root=variable_node._root_,
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
                register_inference(result.bindings[self._id_], self)
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
