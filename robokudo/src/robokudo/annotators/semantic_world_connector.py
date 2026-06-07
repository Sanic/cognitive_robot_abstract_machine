from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from robokudo.utils.transform import get_translation_from_transform_matrix
from robokudo.utils.transform import get_quaternion_from_transform_matrix
from robokudo.utils.annotator_helper import get_cam_to_world_transform_matrix
from robokudo.utils.transform import get_transform_matrix_from_q
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from robokudo.world import world_instance
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from robokudo.types.belief_state import ObjectBeliefState
from scipy.optimize import linear_sum_assignment
from robokudo.types.cv import ImageROI
from robokudo.utils.hypothesis_comparators import ObjectHypothesisComparator
from timeit import default_timer

import numpy as np
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.types.annotation import (
    PoseAnnotation,
    BoundingBox3DAnnotation,
)
from robokudo.types.scene import ObjectHypothesis
from robokudo import world
from semantic_digital_twin.world_description.geometry import Box


class SemanticDigitalTwinConnector(BaseAnnotator):
    """An annotator that synchronizes the current state of the world with the semdt."""

    class Descriptor(BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self) -> None:
                """Initialize a new set of parameters for the descriptor."""

                self.confidence_threshold = 0.05

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "WorldValidator",
        descriptor: "SemanticDigitalTwinConnector.Descriptor" = Descriptor(),
    ) -> None:
        """Default construction. Minimal one-time init!"""
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.object_comparator = (
            ObjectHypothesisComparator(self.get_cas)
            .with_comparator_for(ImageROI, weight=0.2)
            .with_comparator_for(PoseAnnotation, weight=0.4)
            .with_comparator_for(BoundingBox3DAnnotation, weight=0.4)
        )

    def update(self) -> Status:
        """Synchronise the current RoboKudo state with the current semdt state."""
        start_timer = default_timer()

        cas = self.get_cas()
        rk_world = world_instance()

        threshold = self.descriptor.parameters.confidence_threshold

        ohs: list[ObjectHypothesis] = self.get_cas().filter_annotations_by_type(
            ObjectHypothesis
        )

        obs = list(world.get_object_belief_states().values())
        if len(obs) == 0:
            for oh in ohs:
                world.add_object_hypothesis_as_belief_state(oh, cas)
            self.rk_logger.info(
                f"SemDT \nKS Entities: {len(rk_world.kinematic_structure_entities)}\nViews: {len(rk_world.semantic_annotations)}\nConnections: {len(rk_world.connections)}"
            )
            return Status.SUCCESS

        cost_matrix = np.full((len(ohs), len(obs)), 1e9)
        for i, oh in enumerate(ohs):
            for j, ob in enumerate(obs):
                similarity = self.object_comparator.compute_similarity(
                    oh, ob.latest_hypothesis
                )
                cost_matrix[i][j] = -similarity

        hypothesis_indices, instance_indices = linear_sum_assignment(cost_matrix)

        for h_idx, i_idx in zip(hypothesis_indices, instance_indices):
            similarity = -cost_matrix[h_idx, i_idx]
            if similarity > threshold:
                world.update_belief_state_with_object_hypothesis(
                    obs[i_idx], ohs[h_idx], cas
                )
            else:
                world.add_object_hypothesis_as_belief_state(oh, cas)

        self.rk_logger.info(
            f"SemDT \nKS Entities: {len(rk_world.kinematic_structure_entities)}\nViews: {len(rk_world.semantic_annotations)}\nConnections: {len(rk_world.connections)}"
        )
        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
