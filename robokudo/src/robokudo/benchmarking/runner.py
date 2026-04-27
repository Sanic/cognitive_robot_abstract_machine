"""Benchmark runner scaffolding.

This module already includes reusable injection hooks for AnalysisEngine trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors import CrDescriptorFactory


@dataclass
class BenchmarkRunConfig:
    """Configuration for a benchmark run."""

    spec_path: Path
    output_dir: Path


def iter_tree_nodes(root: Any) -> Iterator[Any]:
    """Yield all behavior nodes in a tree in depth-first order."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        children = getattr(node, "children", None)
        if children:
            stack.extend(reversed(children))


def inject_collection_reader_descriptor(
    tree_root: Any,
    camera_config: str,
    camera_config_kwargs: dict[str, Any] | None = None,
    target_annotator_names: list[str] | None = None,
) -> int:
    """Inject a new CollectionReader descriptor into all CollectionReaderAnnotators.

    This enables benchmark-level data-source control while preserving the overall
    AnalysisEngine topology.

    :returns: number of patched CollectionReaderAnnotator nodes
    :raises RuntimeError: if no CollectionReaderAnnotator exists in the tree
    """
    kwargs = camera_config_kwargs or {}
    descriptor = CrDescriptorFactory.create_descriptor(camera_config, **kwargs)
    target_names = set(target_annotator_names or [])
    matched_names: set[str] = set()

    patched = 0
    for node in iter_tree_nodes(tree_root):
        if isinstance(node, CollectionReaderAnnotator):
            node_name = getattr(node, "name", "")
            if target_names and node_name not in target_names:
                continue
            node.descriptor = descriptor
            if node_name:
                matched_names.add(node_name)
            patched += 1

    if patched == 0:
        raise RuntimeError(
            "CollectionReader injection requested, but no CollectionReaderAnnotator "
            "was found in the AnalysisEngine tree."
        )

    if target_names:
        missing = sorted(target_names.difference(matched_names))
        if missing:
            raise RuntimeError(
                "CollectionReader injection target names not found in tree: "
                + ", ".join(missing)
            )
    return patched


def run_benchmark(config: BenchmarkRunConfig) -> None:
    """Run a benchmark.

    This is a scaffold placeholder. Execution integration with RoboKudo pipelines
    will be implemented in follow-up steps.
    """
    raise NotImplementedError(
        "Benchmark execution is not implemented yet. "
        "Use `rk-benchmark validate` for now."
    )
