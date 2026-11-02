from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from semantic_digital_twin.adapters.articulate3d_dataset.exceptions import (
    Articulate3DSceneNotFoundError,
)
from semantic_digital_twin.adapters.articulate3d_dataset.schema import (
    Articulate3DAnnotationFile,
    Articulate3DArticulation,
    Articulate3DPart,
    Articulate3DScene,
)
from semantic_digital_twin.adapters.scannetpp_dataset.loader import (
    ScanNetPPDatasetLoader,
)

# %% loader


@dataclass
class Articulate3DDatasetLoader:
    """
    Loader for the Articulate3D dataset
    (https://huggingface.co/datasets/INSAIT-Institute/Articulate3D), part and
    articulation annotations on the meshes of ScanNet++ scenes.

    .. warning::
        Articulate3D annotates the meshes of ScanNet++ v1. Some scenes' meshes changed
        in later ScanNet++ releases, and their annotations then cover the wrong faces;
        :attr:`scannetpp_dataset_loader` has to read scenes whose meshes match.
    """

    directory: Path
    """
    The directory holding Articulate3D's annotation files.
    """

    scannetpp_dataset_loader: ScanNetPPDatasetLoader
    """
    The loader of the ScanNet++ scenes the annotations refer to.
    """

    def load(self, scene_id: str) -> Articulate3DScene:
        """
        :param scene_id: The ScanNet++ scene's identifier, e.g. ``"f9f95681fd"``.
        :return: The scene with its parts and articulations.
        :raises Articulate3DSceneNotFoundError: if :attr:`directory` holds no
            annotations for the scene.
        """
        parts_path = Articulate3DAnnotationFile.PARTS.path(self.directory, scene_id)
        articulations_path = Articulate3DAnnotationFile.ARTICULATIONS.path(
            self.directory, scene_id
        )
        if not parts_path.is_file() or not articulations_path.is_file():
            raise Articulate3DSceneNotFoundError(
                scene_id=scene_id, directory=self.directory
            )

        articulations = json.loads(articulations_path.read_text())
        return Articulate3DScene(
            scannetpp_scene=self.scannetpp_dataset_loader.load(scene_id),
            parts=Articulate3DPart.all_in_file(parts_path),
            articulations=[
                Articulate3DArticulation.from_json(articulation)
                for articulation in articulations["data"]["articulations"]
            ],
        )
