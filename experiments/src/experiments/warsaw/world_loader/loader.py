"""
The world a scanned scene becomes, and the pictures taken of it.

The scene is loaded as a single body rather than one body per object, because its
objects are labels over shared faces; cutting it into bodies is a later step's work.
What this does beyond loading is draw it: from chosen viewpoints, with chosen objects
painted, which is how a model is shown what it is being asked about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import numpy as np
import trimesh
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.pipeline import (
    CenterLocalGeometryAndPreserveWorldPose,
    Pipeline,
    TransformGeometry,
)
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.inheritance_structure import InheritanceStructureExporter
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from typing_extensions import (
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from experiments.warsaw.painting import PaintColor
from experiments.warsaw.exceptions import (
    BlankRenderError,
    CameraHasNoDirectionError,
    NoSegmentsGivenError,
    SceneBodyNotFoundError,
    WarsawLabelsMisalignedError,
)
from experiments.warsaw.world_loader.scene import (
    LabelSegment,
    PlyPayload,
    WarsawScene,
    segment_label,
)
from experiments.warsaw.world_loader.viewpoints import (
    PictureKind,
    RenderSizes,
    Viewpoint,
    ViewpointChoice,
    changed_pixels,
    is_one_color,
)


@dataclass(frozen=True)
class RenderedPictures:
    """
    Renders of a scene, each with the camera pose it was taken from.
    """

    images: Dict[str, bytes] = field(default_factory=dict)
    """
    The renders as PNG bytes, by the name of each view.
    """

    camera_poses: Dict[str, np.ndarray] = field(default_factory=dict)
    """
    The pose each was rendered from, under the same names.
    """

    field_of_view: Tuple[float, float] = (0.0, 0.0)
    """
    How wide the camera saw, in degrees across and down.
    """

    frame: str = ""
    """
    What the poses are relative to.
    """

    def named(self, naming: Dict[str, str]) -> RenderedPictures:
        """
        Say what each view was written as, so its pose is recorded under that name.

        :param naming: The filename each view was written as, by the name of the view.
        :return: The same pictures, keyed by filename.
        """
        return RenderedPictures(
            images={naming[view]: image for view, image in self.images.items()},
            camera_poses={
                naming[view]: pose
                for view, pose in self.camera_poses.items()
                if view in naming
            },
            field_of_view=self.field_of_view,
            frame=self.frame,
        )


@dataclass
class RenderedSegmentGroup:
    """
    One group of label segments, colored and rendered from every viewpoint.
    """

    index: int
    """
    Which group of the scene's segments this is.
    """

    segments: List[LabelSegment]
    """
    The segments colored in the renders.
    """

    colors: Dict[PrefixedName, PaintColor]
    """
    The color each of them was given.
    """

    images: Dict[str, bytes]
    """
    Per viewpoint, the render taken from it, as PNG bytes.
    """

    camera_poses: Dict[str, np.ndarray] = field(default_factory=dict)
    """
    Per viewpoint, the camera pose it was rendered from.
    """

    @property
    def label(self) -> str:
        """
        :return: The segments' names, short enough for a filename to carry.
        """
        return segment_label(self.segments)


SHARED_FACES_LABEL = "both"
"""
What the legend calls the faces two segments both claim.
"""


@dataclass
class WarsawWorldLoader:
    """
    Load a Warsaw scene directory into a single World, or wrap an existing World for
    rendering and export operations.
    """

    input_directory: Optional[Path] = field(default=None)
    """
    Directory holding the scene to load.

    Can be None if world is provided directly.
    """

    world: Optional[World] = field(default=None)
    """
    Loaded World object.

    Can be provided directly or loaded from input_directory.
    """

    _camera_field_of_view: Tuple[float, float] = field(default=(60, 45))
    """
    Camera field of view for rendering.
    """

    framed_fraction: float = field(default=0.99)
    """
    The share of the scene that has to fall inside each view.

    Below 1 the cameras stand closer and the scene's outermost fringe falls outside the
    view, which on a scanned room crops the stray fragments its walls trail off into
    rather than anything standing in it.
    """

    render_sizes: RenderSizes = field(default_factory=RenderSizes)
    """
    How large the renders are drawn.
    """

    unhighlighted_dimming: float = field(default=0.35)
    """
    How much of its own color the scene keeps where nothing is highlighted.

    The scene is scanned in its own colors, against which a highlight of a similar hue
    is hard to make out. Dimming everything else leaves the room recognizable while the
    highlighted segments are the only part of it in full color.
    """

    scene: Optional[WarsawScene] = field(default=None)
    """
    The scene the world was loaded from, or None for a world provided directly.
    """

    scene_body_name: str = field(default="scene")
    """
    The name of the body carrying the whole scene.
    """

    viewpoint_elevation: float = field(default_factory=lambda: np.radians(30))
    """
    How far above the scene's middle the predefined cameras are lifted, in radians.
    """

    _original_face_colors: Optional[np.ndarray] = field(init=False, default=None)
    """
    The color of each of the scene body's faces before anything was highlighted.
    """

    _renderer_has_drawn: bool = field(init=False, default=False)
    """
    Whether the renderer has handed back a picture with anything in it.

    A driver that refuses to draw into a hidden window refuses from the first picture
    onwards, so the renders are checked until one of them proves it draws and taken as
    they come afterwards.
    """

    _plain_views: Dict[Tuple[str, ...], Dict[str, bytes]] = field(
        init=False, default_factory=dict
    )
    """
    The scene rendered dimmed and unhighlighted, kept so it is rendered once rather than
    once per region measured against it.
    """

    def __post_init__(self):
        if self.world is None and self.input_directory is None:
            raise ValueError("Either input_directory or world must be provided")

        if self.world is None:
            if self.scene is None:
                self.scene = WarsawScene.from_directory(self.input_directory)
            self.world = self._world_from_scene(self.scene, self.scene_body_name)
            self._verify_faces_carry_the_labels()

        # Face colors are read once here, while the mesh still carries the colors it was
        # scanned in, because highlighting replaces them.
        if self.world.bodies_with_collision:
            self._original_face_colors = np.asarray(
                self.scene_mesh.visual.face_colors
            ).copy()
            # The body reads the scan's own file, so this mesh carries that file the way
            # the scene's mesh did. Every render cuts a piece out of it and trimesh
            # copies a mesh's metadata into every piece it cuts, which on a scan is a few
            # hundred megabytes per picture. A world given directly may hold a body read
            # from a format that keeps no such payload, so there may be none to let go of.
            self.scene_mesh.metadata.pop(PlyPayload.RAW.value, None)

    @staticmethod
    def _world_from_scene(scene: WarsawScene, scene_body_name: str = "scene") -> World:
        """
        Build the world a scene describes: its whole mesh, under a single root.

        The scene stays one body rather than one body per object, because its objects
        are labels over shared faces: a drawer front is part of the drawer and of the
        cabinet holding it, and cutting the mesh into bodies would have to duplicate
        such geometry or take it from one of them.

        The body reads the scene's file directly, so its faces arrive in the order the
        labels were written for.

        :param scene: The scene to build a world from.
        :param scene_body_name: What to call the body carrying the whole scene.
        :return: A world holding the scene under a single root.
        """
        main_world = World()
        root = Body(name=PrefixedName("root_body"))
        shapes = ShapeCollection([Mesh.from_file(str(scene.mesh_path))])
        scene_body = Body(
            name=PrefixedName(scene_body_name), collision=shapes, visual=shapes
        )
        with main_world.modify_world():
            main_world.add_body(root)
            main_world.add_body(scene_body)
            main_world.add_connection(
                FixedConnection(
                    parent=root,
                    child=scene_body,
                    name=PrefixedName(f"root_to_{scene_body_name}"),
                )
            )

        pipeline = Pipeline(
            steps=[
                TransformGeometry(scene.world_T_source),
                CenterLocalGeometryAndPreserveWorldPose(),
            ]
        )
        return pipeline.apply(main_world)

    def _verify_faces_carry_the_labels(self) -> None:
        """
        Check that the world was built from the faces the labels were written for.

        :raises WarsawLabelsMisalignedError: If it was built from other faces.
        """
        loaded_faces = self.scene_mesh.faces
        if not np.array_equal(loaded_faces, self.scene.mesh.faces):
            raise WarsawLabelsMisalignedError(
                scene_mesh=self.scene.mesh_path,
                labelled_faces=len(self.scene.mesh.faces),
                loaded_faces=len(loaded_faces),
            )

    # %% the scene's body and its labelled objects

    @cached_property
    def scene_body(self) -> Body:
        """
        :return: The body carrying the scene's geometry.
        """
        for body in self.world.bodies_with_collision:
            if body.name.name == self.scene_body_name:
                return body
        raise SceneBodyNotFoundError(
            scene_body_name=self.scene_body_name,
            body_names=[body.name.name for body in self.world.bodies_with_collision],
        )

    @property
    def scene_mesh(self) -> trimesh.Trimesh:
        """
        :return: The scene body's mesh, as the world holds it.
        """
        return self.scene_body.collision[0].mesh

    @cached_property
    def label_segments(self) -> List[LabelSegment]:
        """
        :return: Every object the scene labels, in class order.
        """
        return list(self.scene.segments())

    def segments_of_classes(self, class_names: Iterable[str]) -> List[LabelSegment]:
        """
        :param class_names: The classes to take the segments of.
        :return: The scene's segments of those classes, in class order.
        """
        wanted = set(class_names)
        return [
            segment for segment in self.label_segments if segment.class_name in wanted
        ]

    # %% drawing the scene

    def export_semantic_annotation_inheritance_structure(
        self, output_directory: Path
    ) -> None:
        """
        Export kinematic structure and semantic annotations to JSON files.

        :param output_directory: Directory to write JSON files to.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        self.world.export_kinematic_structure_tree_to_json(
            output_directory / "kinematic_structure.json",
            include_connections=False,
        )
        InheritanceStructureExporter(
            SemanticAnnotation, output_directory / "semantic_annotations.json"
        ).export()

    def label_segment_groups(
        self,
        group_size: int = 8,
        segments: Optional[Iterable[LabelSegment]] = None,
        headless: bool = False,
        pictured: bool = True,
    ) -> Iterator[RenderedSegmentGroup]:
        """
        Walk the scene's objects a group at a time, colouring and drawing each group.

        The scene is left in its own colors once the last group has been walked.

        :param group_size: How many segments to take at once.
        :param segments: The segments to walk through, defaulting to all of the scene's.
        :param headless: Whether to render without opening a window.
        :param pictured: Whether to draw each group. Not drawing it is what a run asking
            from the text alone needs, and it is where such a run saves its time:
            nothing is painted, no camera is placed, and the group carries no colour,
            since a colour names nothing in a picture that was never made.
        :return: One group at a time.
        """
        segments = list(self.label_segments if segments is None else segments)
        if not pictured:
            for index, start in enumerate(range(0, len(segments), group_size)):
                yield RenderedSegmentGroup(
                    index=index,
                    segments=segments[start : start + group_size],
                    colors={},
                    images={},
                )
            return

        camera_poses = self.compute_camera_poses()
        try:
            for index, start in enumerate(range(0, len(segments), group_size)):
                group = segments[start : start + group_size]
                colors = self._apply_highlight_to_segments(group)
                yield RenderedSegmentGroup(
                    index=index,
                    segments=group,
                    colors=colors,
                    images=self.render_scene_from_camera_poses(
                        camera_poses, headless=headless
                    ),
                    camera_poses=camera_poses,
                )
        finally:
            self._reset_segment_colors()

    def export_scene_to_pngs(
        self,
        group_size: int,
        output_directory: Path,
        headless: bool = False,
    ) -> None:
        """
        Export rendered images of the scene with highlighted groups of label segments.

        :param group_size: Number of label segments to highlight in each group.
        :param output_directory: Directory to write images to.
        :param headless: Whether to render without opening a window.
        """
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        for pose_name, image in self.render_scene_from_camera_poses(
            self.compute_camera_poses(), headless=headless
        ).items():
            (output_directory / f"original_{pose_name}.png").write_bytes(image)

        for group in self.label_segment_groups(
            group_size=group_size, headless=headless
        ):
            for pose_name, image in group.images.items():
                (
                    output_directory
                    / f"group_{group.index}_{pose_name}__{group.label}.png"
                ).write_bytes(image)

    def render_scene_from_predefined_poses(
        self, output_path: Path, filename_prefix: str, headless: bool = False
    ) -> None:
        """
        Render the world from each of the predefined camera poses, writing one image per
        pose.

        :param output_path: Directory to save images.
        :param filename_prefix: Prefix for image filenames.
        :param headless: Whether to render without opening a window.
        """
        output_path = Path(output_path)
        for pose_name, image in self.render_scene_from_camera_poses(
            self.compute_camera_poses(), headless=headless
        ).items():
            (output_path / f"{filename_prefix}_{pose_name}.png").write_bytes(image)

    def render_scene_from_camera_poses(
        self,
        camera_poses: Dict[str, np.ndarray],
        headless: bool = False,
    ) -> Dict[str, bytes]:
        """
        Render the world from several camera poses at once.

        The scene is built once for all of them, which on a scanned room is most of the
        work a render costs.

        :param camera_poses: The poses to render from, by the name of each viewpoint.
        :param headless: Whether to render without opening a window.
        :return: Per viewpoint, its render as PNG bytes.
        """
        return self._render_from_poses(self._render_scene(), camera_poses, headless)

    def _render_from_poses(
        self,
        scene: trimesh.Scene,
        camera_poses: Dict[str, np.ndarray],
        headless: bool = False,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, bytes]:
        """
        Render one scene from several camera poses.

        :param scene: The scene to render.
        :param camera_poses: The poses to render from, by the name of each viewpoint.
        :param headless: Whether to render without opening a window.
        :param resolution: How large to render, defaulting to the size a picture is kept
            at. A render made only to be measured and thrown away costs what its pixels
            cost and needs none of them.
        :return: Per viewpoint, its render as PNG bytes.
        """
        scene.camera.fov = self._camera_field_of_view
        images = {}
        for pose_name, camera_pose in camera_poses.items():
            scene.graph[scene.camera.name] = camera_pose
            images[pose_name] = scene.save_image(
                resolution=resolution or self.render_sizes.kept,
                visible=not headless,
                # A scan is a single layer of surface whose faces point whichever way
                # they were reconstructed, so culling the ones facing away hides parts of
                # the room for no reason a viewer could tell -- a drawer front vanishes
                # while the handle on it stays.
                flags={"cull": False},
            )
            self._verify_the_renderer_draws(pose_name, images[pose_name])
        return images

    def _verify_the_renderer_draws(self, viewpoint: str, render: bytes) -> None:
        """
        Check that the renderer draws at all, until one render proves that it does.

        A blank render fails nothing on its own: what follows measures no difference
        between two of them and puts them to a model as though they showed something, so
        a run against a renderer that draws nothing costs what a real one costs and
        concludes whatever the model says about black pictures.

        :param viewpoint: The viewpoint the render was taken from.
        :param render: The render, as PNG bytes.
        :raises BlankRenderError: If the renderer has not yet drawn and this render is a
            single flat color.
        """
        if self._renderer_has_drawn:
            return
        if is_one_color(render):
            raise BlankRenderError(viewpoint=viewpoint)
        self._renderer_has_drawn = True

    @cached_property
    def _ray_tracer(self) -> RayTracer:
        """
        :return: The ray tracer whose scene the renders are taken from.

        It holds the world's own meshes rather than copies of them, so it keeps up with
        recoloring on its own and is built once instead of once per image.
        """
        return RayTracer(world=self.world)

    def _render_scene(self) -> trimesh.Scene:
        """
        :return: The scene to render, brought up to date with the world.
        """
        self._ray_tracer.update_scene()
        return self._ray_tracer.scene

    # %% where to put the camera

    @cached_property
    def _scene_points(self) -> np.ndarray:
        """
        :return: Every vertex the scene draws, in the world's frame.
        """
        return np.concatenate(
            [
                shape.mesh_in_frame(self.world.root).vertices
                for body in self.world.bodies_with_collision
                for shape in body.collision
            ]
        )

    @staticmethod
    def _looking_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Place a camera at *eye* facing *target*.

        :param eye: Where the camera stands.
        :param target: What it faces.
        :return: The camera's transform in the world's frame.
        """
        forward = target - eye
        if not np.linalg.norm(forward):
            raise CameraHasNoDirectionError(eye=eye.tolist(), target=target.tolist())
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        if not np.linalg.norm(right):
            raise CameraHasNoDirectionError(eye=eye.tolist(), target=target.tolist())
        right = right / np.linalg.norm(right)

        transform = np.eye(4)
        transform[:3, 0] = right
        transform[:3, 1] = np.cross(right, forward)
        # A camera looks down its own negative z.
        transform[:3, 2] = -forward
        transform[:3, 3] = eye
        return transform

    def compute_camera_poses(
        self, points: Optional[np.ndarray] = None
    ) -> Dict[Viewpoint, np.ndarray]:
        """
        :param points: The points to frame, defaulting to the whole scene's. Framing a
            few segments instead is what makes a handle of two hundred faces visible at
            all, rather than three pixels of a picture of a room.
        :return: One camera pose per named viewpoint, each standing just far enough from
            those points for :attr:`framed_fraction` of them to fall inside the view.

        The distance is measured from what is framed rather than fixed, so a scene of any
        size is framed rather than viewed from within itself.
        """
        framed = self._scene_points if points is None else points
        middle = (framed.min(axis=0) + framed.max(axis=0)) / 2

        poses = {}
        for viewpoint in Viewpoint:
            direction = np.array(
                [
                    np.cos(self.viewpoint_elevation) * np.cos(viewpoint.azimuth),
                    np.cos(self.viewpoint_elevation) * np.sin(viewpoint.azimuth),
                    np.sin(self.viewpoint_elevation),
                ]
            )
            distance = self._framing_distance(framed - middle, direction)
            poses[viewpoint] = self._looking_at(middle + distance * direction, middle)
        return poses

    def _framing_distance(self, points: np.ndarray, direction: np.ndarray) -> float:
        """
        Measure how far along *direction* a camera has to stand for
        :attr:`framed_fraction` of the points to fall inside its view.

        A point's room in the view grows with its distance from the camera, so one on the
        near side needs more room than one equally far off the view's axis on the far
        side. Each point is therefore solved for separately and the furthest answer wins.

        ..note:: This measures the scene's own geometry rather than the corners of the box
            around it, which stand in empty air and would push the camera needlessly far
            back.

        :param points: The scene's points, relative to what the camera faces.
        :param direction: The direction from the scene to the camera.
        :return: The distance from the scene to stand at.
        """
        orientation = self._looking_at(direction, np.zeros(3))
        right, up = orientation[:3, 0], orientation[:3, 1]
        horizontal, vertical = (
            np.tan(np.radians(angle) / 2) for angle in self._camera_field_of_view
        )

        # A point at depth ``distance - along`` fits when its offset across the view is
        # within that depth's half-width, so it needs ``along + offset / half_angle``.
        along = points @ direction
        needed = np.concatenate(
            [
                along + np.abs(points @ right) / horizontal,
                along + np.abs(points @ up) / vertical,
            ]
        )
        return float(np.percentile(needed, self.framed_fraction * 100))

    # %% painting the labelled objects

    def _apply_highlight_to_faces(
        self, highlights: Sequence[Tuple[Color, np.ndarray]]
    ) -> None:
        """
        Paint sets of the scene's faces, dimming everything else.

        :param highlights: The color to paint each set of faces in. Later sets paint
            over earlier ones where they overlap.
        """
        face_colors = self._dimmed_face_colors.copy()
        for color, faces in highlights:
            face_colors[faces] = trimesh.visual.color.to_rgba(color.to_rgba())
        self.scene_mesh.visual.face_colors = face_colors

    def _apply_highlight_to_segments(
        self, segments: Iterable[LabelSegment]
    ) -> Dict[PrefixedName, PaintColor]:
        """
        Color a group of the scene's objects, dimming everything else.

        :param segments: The segments to color.
        :return: The color each of them was given.
        :raises TooManyToPaintError: If there are more segments than colors to tell them
            apart by.
        """
        segments = list(segments)
        colors = PaintColor.distinct(len(segments))
        self._apply_highlight_to_faces(
            [
                (color.color, segment.face_indices)
                for segment, color in zip(segments, colors)
            ]
        )
        return {segment.name: color for segment, color in zip(segments, colors)}

    @cached_property
    def _dimmed_face_colors(self) -> np.ndarray:
        """
        :return: The scene's own face colors, dimmed by :attr:`unhighlighted_dimming`.

        Every group of highlights is painted over these, so they are dimmed once rather
        than once per group, which on a scanned room is seconds of work each time.
        """
        face_colors = self._original_face_colors.copy()
        face_colors[:, :3] = (
            face_colors[:, :3].astype(np.float64) * self.unhighlighted_dimming
        ).astype(np.uint8)
        return face_colors

    def _reset_segment_colors(self) -> None:
        """
        Paint the scene back in the colors it was scanned in.
        """
        if self._original_face_colors is not None:
            self.scene_mesh.visual.face_colors = self._original_face_colors.copy()

    def pair_highlights(
        self, one: LabelSegment, other: LabelSegment
    ) -> Tuple[List[Tuple[Color, np.ndarray]], Dict[str, Color]]:
        """
        Work out how to color two segments so that what they disagree about can be seen.

        A face two segments both claim can only take one color, so a picture painted in
        two colors shows an overlap as though it belonged to whichever was painted last.
        The faces both claim therefore take a third color, which is the only way the
        question being asked is visible in the picture at all.

        :param one: One of the segments.
        :param other: The other segment.
        :return: What to paint, and what each color stands for: each segment's name for
            the faces it alone claims, and ``both`` for the faces they share, which is
            absent when they share none.
        """
        shared = np.intersect1d(
            one.face_indices, other.face_indices, assume_unique=True
        )
        colors = Color.distinct_colors(3)
        highlights = [
            (
                colors[0],
                np.setdiff1d(one.face_indices, other.face_indices, assume_unique=True),
            ),
            (
                colors[1],
                np.setdiff1d(other.face_indices, one.face_indices, assume_unique=True),
            ),
            (colors[2], shared),
        ]

        legend = {str(one.name): colors[0], str(other.name): colors[1]}
        if len(shared):
            legend[SHARED_FACES_LABEL] = colors[2]
        return highlights, legend

    def highlight_pair(
        self, one: LabelSegment, other: LabelSegment
    ) -> Dict[str, Color]:
        """
        Color two segments so that what they disagree about can be seen.

        :param one: One of the segments.
        :param other: The other segment.
        :return: What each color stands for, as :meth:`pair_highlights` describes it.
        """
        highlights, legend = self.pair_highlights(one, other)
        self._apply_highlight_to_faces(highlights)
        return legend

    def face_indices_of(self, segments: Iterable[LabelSegment]) -> np.ndarray:
        """
        :param segments: The segments to take the faces of.
        :return: The index of every face they cover.
        :raises NoSegmentsGivenError: If no segments were given.
        """
        gathered = [segment.face_indices for segment in segments]
        if not gathered:
            raise NoSegmentsGivenError()
        return np.concatenate(gathered)

    @property
    def world_T_scene_mesh(self) -> np.ndarray:
        """
        :return: Where the world places the coordinates :attr:`scene_mesh` is written
            in.

        Loading centres the scene body's geometry on itself and carries the difference
        in the body's pose, so a point read straight off that mesh is not where the
        world holds it, and a camera placed from one is not standing where it says.
        """
        shape = self.scene_body.collision[0]
        return self.world.transform(shape.origin, self.world.root).to_np()

    def points_of(self, segments: Iterable[LabelSegment]) -> np.ndarray:
        """
        :param segments: The segments to take the geometry of.
        :return: The vertices their faces are drawn from, in the world's frame.
        :raises NoSegmentsGivenError: If no segments were given.
        """
        faces = self.face_indices_of(segments)
        return trimesh.transform_points(
            self.scene_mesh.vertices[self.scene_mesh.faces[faces].ravel()],
            self.world_T_scene_mesh,
        )

    def poses_framing(
        self,
        segments: Iterable[LabelSegment],
        viewpoints: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        :param segments: The segments a camera is to frame.
        :param viewpoints: Which named viewpoints to stand at, defaulting to all of
            them.
        :return: Where the camera stands for each of them, in the world's frame.
        :raises NoSegmentsGivenError: If no segments were given.
        """
        return self._chosen_viewpoints(
            self.compute_camera_poses(self.points_of(segments)), viewpoints
        )

    def segments_alone(
        self, segments: Iterable[LabelSegment], face_colors: np.ndarray
    ) -> trimesh.Trimesh:
        """
        Cut some segments' geometry out of the scene, leaving the rest of it out.

        :param segments: The segments to cut out.
        :param face_colors: The color of every face of the whole scene, of which the
            segments' own are taken.
        :return: Their geometry by itself, where the world holds it.
        :raises NoSegmentsGivenError: If no segments were given.
        """
        faces = np.unique(self.face_indices_of(segments))
        alone = self.scene_mesh.submesh([faces], append=True)
        alone.visual.face_colors = face_colors[faces]
        alone.apply_transform(self.world_T_scene_mesh)
        return alone

    def segments_as_scanned(
        self, segments: Iterable[LabelSegment]
    ) -> Union[trimesh.Trimesh, trimesh.Scene]:
        """
        Build what a plain picture of some segments draws: them, looking as they look.

        Where the scene ships an appearance mesh, that is what they look like -- the
        source's own texture and normals, over the same faces. Where it ships none, as
        every scan does, it is the mesh's own colours, which is what a plain picture has
        always been.

        :param segments: The segments to draw.
        :return: The geometry to draw, as one mesh or as the pieces a texture comes in.
        """
        segments = list(segments)
        if self.scene is None or self.scene.appearance is None:
            return self.segments_alone(segments, self._original_face_colors)
        return self.scene.appearance.faces_alone(
            np.concatenate([one.face_indices for one in segments])
            if segments
            else np.array([], dtype=int)
        )

    def render_segments_as_scanned(
        self,
        segments: Iterable[LabelSegment],
        viewpoints: Optional[Sequence[str]] = None,
        headless: bool = False,
    ) -> RenderedPictures:
        """
        Draw some segments by themselves, looking as they look.

        :param segments: The segments to draw.
        :param viewpoints: Which named viewpoints to render, defaulting to all of them.
        :param headless: Whether to render without opening a window.
        :return: Per viewpoint, its render and the pose it was taken from.
        """
        segments = list(segments)
        drawn = self.segments_as_scanned(segments)
        poses = self.poses_framing(segments, viewpoints)
        scene = drawn if isinstance(drawn, trimesh.Scene) else trimesh.Scene(drawn)
        return self._pictures(self._render_from_poses(scene, poses, headless), poses)

    def render_segments_alone(
        self,
        segments: Iterable[LabelSegment],
        face_colors: np.ndarray,
        viewpoints: Optional[Sequence[str]] = None,
        headless: bool = False,
    ) -> RenderedPictures:
        """
        Render some segments' geometry by itself, with the rest of the scene left out.

        Framing a camera closely on a few hundred faces inside a scanned room puts it
        behind the room's own walls and cabinet fronts, so a close-up of the whole scene
        shows whatever stands in the way rather than what was asked about. Rendering the
        segments alone is what makes them visible; where they sit in the room is what a
        context view is for.

        :param segments: The segments to render.
        :param face_colors: The color of every face of the whole scene, of which the
            segments' own are taken.
        :param viewpoints: Which named viewpoints to render, defaulting to all of them.
        :param headless: Whether to render without opening a window.
        :return: Per viewpoint, its render and the pose it was taken from.
        """
        segments = list(segments)
        alone = self.segments_alone(segments, face_colors)
        poses = self.poses_framing(segments, viewpoints)
        return self._pictures(
            self._render_from_poses(trimesh.Scene(alone), poses, headless), poses
        )

    @property
    def camera_field_of_view(self) -> Tuple[float, float]:
        """
        :return: How wide the camera sees, in degrees across and down.
        """
        return self._camera_field_of_view

    def _pictures(
        self,
        images: Dict[str, bytes],
        camera_poses: Dict[str, np.ndarray],
    ) -> RenderedPictures:
        """
        Put renders together with the poses they were taken from.

        :param images: The renders, by the name of each view.
        :param camera_poses: The pose each was taken from, under the same names.
        :return: The two together, with what else is needed to use a pose.
        """
        return RenderedPictures(
            images=images,
            camera_poses=camera_poses,
            field_of_view=self.camera_field_of_view,
            frame=str(self.world.root.name),
        )

    def _chosen_viewpoints(
        self,
        poses: Dict[str, np.ndarray],
        viewpoints: Optional[Sequence[str]],
    ) -> Dict[str, np.ndarray]:
        """
        :param poses: The poses of every named viewpoint.
        :param viewpoints: The names to keep, or None to keep all of them.
        :return: The poses that were asked for.
        """
        if viewpoints is None:
            return poses
        wanted = set(viewpoints)
        return {name: pose for name, pose in poses.items() if name in wanted}

    def render_region(
        self,
        segments: Iterable[LabelSegment],
        highlights: Sequence[Tuple[Color, np.ndarray]],
        viewpoints: Optional[Sequence[str]] = None,
        headless: bool = False,
        choose_viewpoint: Optional[ViewpointChoice] = None,
        context_segments: Optional[Iterable[LabelSegment]] = None,
    ) -> RenderedPictures:
        """
        Render a part of the scene the three ways a question about it needs answering.

        The close-up shows what is being asked about, on its own so that nothing stands
        in front of it; the context view shows where in the room it is; and the plain
        close-up shows the same geometry in the colors it was scanned in, without which a
        viewer is judging paint rather than an object.

        :param segments: The segments the close-ups show.
        :param highlights: The colors to paint, as :meth:`_apply_highlight_to_faces`
            takes them.
        :param viewpoints: Which named viewpoints to render, defaulting to all of them.
        :param headless: Whether to render without opening a window.
        :param context_segments: What the context view is framed on, defaulting to the
            whole scene. Framing it on a segment together with its measured neighbours
            is what makes a small object visible in it at all: a mug fills 0.00% of a
            picture of the room and 9.6% of a picture of the counter it stands on.
        :param choose_viewpoint: How to keep only the viewpoint that shows the most,
            measured rather than chosen: a cabinet standing against a wall is invisible
            from three of the four, and a fixed viewpoint then hands a viewer a picture
            of a room with nothing marked in it. ``in-room`` measures it in the scene, so
            that what stands in front of the segments counts against a viewpoint;
            ``alone`` measures it on the segments by themselves, which is what a hundred
            of these can afford. None keeps every viewpoint.
        :return: The renders and the pose each was taken from, keyed
            ``<kind>_<viewpoint>``.
        """
        segments = list(segments)
        frame = None if context_segments is None else self.points_of(context_segments)
        if choose_viewpoint == ViewpointChoice.IN_ROOM:
            viewpoints = (
                self.viewpoint_showing_all(segments, viewpoints, headless, frame),
            )
        elif choose_viewpoint == ViewpointChoice.ALONE:
            viewpoints = (
                self.viewpoint_showing_all_alone(segments, viewpoints, headless),
            )
        elif choose_viewpoint not in (None, ViewpointChoice.ALL):
            raise ValueError(
                f"{choose_viewpoint!r} is no way of choosing a viewpoint; it is one of "
                f"{', '.join(one.value for one in ViewpointChoice)}"
            )

        self._apply_highlight_to_faces(highlights)
        painted = np.asarray(self.scene_mesh.visual.face_colors).copy()
        closeups = self.render_segments_alone(segments, painted, viewpoints, headless)
        context_poses = self._chosen_viewpoints(
            self.compute_camera_poses(frame), viewpoints
        )
        contexts = self._pictures(
            self._render_from_poses(self._render_scene(), context_poses, headless),
            context_poses,
        )

        self._reset_segment_colors()
        plains = self.render_segments_as_scanned(segments, viewpoints, headless)

        of_each_kind = (
            (PictureKind.CLOSEUP, closeups),
            (PictureKind.CONTEXT, contexts),
            (PictureKind.PLAIN, plains),
        )
        return self._pictures(
            images={
                f"{kind}_{name}": image
                for kind, pictures in of_each_kind
                for name, image in pictures.images.items()
            },
            camera_poses={
                f"{kind}_{name}": pose
                for kind, pictures in of_each_kind
                for name, pose in pictures.camera_poses.items()
            },
        )

    def viewpoint_showing_all(
        self,
        segments: Iterable[LabelSegment],
        viewpoints: Optional[Sequence[str]] = None,
        headless: bool = False,
        frame: Optional[np.ndarray] = None,
    ) -> str:
        """
        Find the viewpoint from which the least-visible of some segments is seen best.

        Each segment is measured on its own, and the viewpoints are ranked by their
        *worst* segment rather than by their total. Ranking by the total lets a large
        object decide for a small one: a door of thirty thousand faces and the handle on
        it are seen best from opposite sides, and the sum picks the side showing the
        whole door and none of the handle -- which is the object whose membership was in
        question. Whichever segment is hardest to see is the one that decides.

        :param segments: The segments that all have to be visible.
        :param viewpoints: Which named viewpoints to consider, defaulting to all.
        :param headless: Whether to render without opening a window.
        :param frame: The points the views are framed on, defaulting to the whole scene.
            The choice has to be made at the framing it is made for, since which
            viewpoint shows an object best depends on how closely it is framed.
        :return: The name of the viewpoint that shows the least-visible segment best.
        """
        segments = list(segments)
        if not segments:
            raise NoSegmentsGivenError()
        poses = self._chosen_viewpoints(self.compute_camera_poses(frame), viewpoints)
        rooms = self._dimmed_views(poses, headless, cacheable=frame is None)
        color = Color.distinct_colors(1)[0]

        worst = {name: None for name in poses}
        for segment in segments:
            self._apply_highlight_to_faces([(color, segment.face_indices)])
            shown = self._render_from_poses(self._render_scene(), poses, headless)
            for name, image in shown.items():
                visible = changed_pixels(image, rooms[name])
                if worst[name] is None or visible < worst[name]:
                    worst[name] = visible
        self._reset_segment_colors()
        return max(worst, key=lambda name: worst[name])

    def presented_area(self, segment: LabelSegment, direction: np.ndarray) -> float:
        """
        Measure how much of a segment turns towards a direction, without drawing it.

        How large a face looks from a direction is its area foreshortened by how squarely
        it faces -- ``area * |n . d|`` summed over the segment. It is arithmetic over
        arrays the mesh already carries, where a render is a tenth of a second.

        Which way a face points does not enter into it, because the renders are made
        without culling: a scan is a single layer of surface whose faces point whichever
        way they were reconstructed, so one pointing away is drawn exactly like one
        pointing towards. Counting only the faces whose normals face the camera measures
        a picture nobody is taking, and picks a different viewpoint than the render does.

        ..note:: It knows nothing of what stands in front of the segment, nor of the
            segment standing in front of itself. It ranks; it does not decide.

        :param segment: The segment to measure.
        :param direction: The direction it is looked at from, from the camera outwards.
        :return: The area of it presented to that direction, in square metres.
        """
        mesh = self.scene_mesh
        towards = np.abs(mesh.face_normals[segment.face_indices] @ direction)
        return float(towards @ mesh.area_faces[segment.face_indices])

    def viewpoint_showing_all_alone(
        self,
        segments: Iterable[LabelSegment],
        viewpoints: Optional[Sequence[str]] = None,
        headless: bool = False,
        considered: int = len(Viewpoint),
    ) -> str:
        """
        Choose the viewpoint showing the least-visible of some segments best, measured
        on the segments by themselves rather than in the room.

        The same choice as :meth:`viewpoint_showing_all` and far cheaper. Every viewpoint
        is first ranked by :meth:`presented_area`, which draws nothing at all; only the
        best few are then drawn, and drawn small, since what is wanted of those renders
        is which of them shows more and not a picture anyone keeps.

        What it gives up against :meth:`viewpoint_showing_all` is occlusion by everything
        else -- a cabinet standing in front of the pair no longer counts against a
        viewpoint -- which is the right trade when the picture it is chosen for shows the
        segments alone.

        :param segments: The segments that all have to be visible.
        :param viewpoints: Which named viewpoints to consider, defaulting to all.
        :param headless: Whether to render without opening a window.
        :param considered: How many of the ranked viewpoints to draw and compare, by
            default all of them. Drawing fewer saved four percent of the time on this
            scene and picked a different viewpoint than drawing all four in two cases of
            six, the renders having never been the cost they looked like; the ranking is
            left for a caller who has more viewpoints than four to choose between. One
            skips drawing altogether and trusts the arithmetic, which cannot see a
            segment hiding behind itself.
        :return: The name of the viewpoint that shows the least-visible segment best.
        """
        segments = list(segments)
        faces = np.unique(self.face_indices_of(segments))
        poses = self.poses_framing(segments, viewpoints)

        # Ranked by the segment each viewpoint shows worst, so that a handle is not
        # outvoted by the door it is screwed to.
        middle = self.points_of(segments).mean(axis=0)
        ranked = sorted(
            poses,
            key=lambda name: min(
                self.presented_area(
                    segment,
                    (middle - poses[name][:3, 3])
                    / np.linalg.norm(middle - poses[name][:3, 3]),
                )
                for segment in segments
            ),
            reverse=True,
        )
        looked_at = {name: poses[name] for name in ranked[: max(considered, 1)]}
        if len(looked_at) == 1:
            return next(iter(looked_at))

        dimmed = self._dimmed_face_colors[faces]
        alone = self.segments_alone(segments, self._dimmed_face_colors)
        rooms = self._render_from_poses(
            trimesh.Scene(alone), looked_at, headless, self.render_sizes.deciding
        )

        highlight = trimesh.visual.color.to_rgba(Color.distinct_colors(1)[0].to_rgba())
        worst: Dict[str, Optional[int]] = {name: None for name in looked_at}
        for segment in segments:
            painted = dimmed.copy()
            painted[np.searchsorted(faces, segment.face_indices)] = highlight
            alone.visual.face_colors = painted
            shown = self._render_from_poses(
                trimesh.Scene(alone), looked_at, headless, self.render_sizes.deciding
            )
            for name, image in shown.items():
                visible = changed_pixels(image, rooms[name])
                if worst[name] is None or visible < worst[name]:
                    worst[name] = visible
        return max(worst, key=lambda name: worst[name])

    def _dimmed_views(
        self,
        poses: Dict[str, np.ndarray],
        headless: bool,
        cacheable: bool,
    ) -> Dict[str, bytes]:
        """
        Render the scene dimmed and unhighlighted, from given poses.

        A context view is measured against this rather than against the scene in its own
        colors, because a highlight render dims everything else: measured against the
        undimmed scene every pixel differs, and the measurement reports the dimming
        instead of the highlight.

        :param poses: The camera poses to render from.
        :param headless: Whether to render without opening a window.
        :param cacheable: Whether these poses are the whole scene's, which every region
            shares and which is therefore worth rendering once.
        :return: Per viewpoint, the scene dimmed and unhighlighted.
        """
        key = tuple(sorted(poses))
        if cacheable and key in self._plain_views:
            return self._plain_views[key]

        self._apply_highlight_to_faces([])
        views = self._render_from_poses(self._render_scene(), poses, headless)
        self._reset_segment_colors()
        if cacheable:
            self._plain_views[key] = views
        return views
