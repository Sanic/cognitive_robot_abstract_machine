"""
Which Articulate3D scenes are tuned on and which are evaluated.

The validation scenes stay locked until the code, the ontology and the protocol are
frozen, so that nothing tuned can have seen them: a validation scene is converted, run
or scored only from a checkout with no local changes whose commit carries a freezing
tag.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration

from experiments.warsaw.articulate3d.exceptions import ValidationSceneLockedError
from experiments.warsaw.pipeline.provenance import inspect_source

# %% the splits


class SceneSplit(PathEnumeration):
    """
    The scene lists of Articulate3D's splits, one scene id per line.
    """

    TRAIN = Path(__file__).resolve().parent / "splits" / "train.txt"
    """
    The scenes prompts, settings, the ontology and the protocol are tuned on.
    """

    VALIDATION = Path(__file__).resolve().parent / "splits" / "val.txt"
    """
    The scenes the frozen pipeline is evaluated on, once.
    """

    @property
    def scene_ids(self) -> frozenset[str]:
        """
        :return: The scenes the split holds.
        """
        return frozenset(self.read_text().split())


# %% the lock on the validation scenes


@dataclass(frozen=True)
class ValidationLock:
    """
    Refuses every validation scene unless the checkout is frozen: it has no local
    changes, and its commit carries a tag whose name starts with :attr:`tag_prefix`.
    """

    repository: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[5]
    )
    """
    The checkout the evaluation runs from.
    """

    tag_prefix: str = "articulate3d-frozen-"
    """
    How the name of a tag freezing the code starts.
    """

    def check(self, scene_id: str) -> None:
        """
        :param scene_id: A scene about to be converted, run or scored.
        :raises ValidationSceneLockedError: If it is a validation scene and the checkout
            is not frozen.
        """
        if scene_id not in SceneSplit.VALIDATION.scene_ids:
            return
        source, _ = inspect_source(self.repository)
        tags = (
            []
            if source.commit is None
            else subprocess.run(
                ["git", "tag", "--points-at", source.commit],
                cwd=self.repository,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        )
        frozen = any(tag.startswith(self.tag_prefix) for tag in tags)
        if source.dirty is False and frozen:
            return
        raise ValidationSceneLockedError(
            scene_id=scene_id,
            commit=source.commit,
            local_changes=source.status,
            tags=tags,
            tag_prefix=self.tag_prefix,
        )
