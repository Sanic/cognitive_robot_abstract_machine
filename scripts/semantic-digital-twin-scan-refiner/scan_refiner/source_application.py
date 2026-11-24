"""Reviewed source copies, live-world reload checks and labelled geometry exports."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path
from krrood.patterns.path_enumeration import PathEnumeration
from importlib import import_module
import hashlib
import json
import shutil
import tarfile
from uuid import uuid4
from xml.etree import ElementTree
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from typing_extensions import TYPE_CHECKING
from krrood.adapters.json_serializer import DataclassJSONSerializer
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    FileUriResolver,
    PackageUriResolver,
)
from semantic_digital_twin.world_source import WorldSource
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Mesh,
    MeshFileFormat,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from scan_refiner.dimensions import BodyDimensions
from scan_refiner.refinement import BodyTranslation
from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole
from scan_refiner.world import WorldSnapshot
from scan_refiner.exceptions import InvalidEvidence

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% Artifact vocabulary
class SourceArtifact(PathEnumeration):
    """Files published together after successful source reload validation."""

    REVIEWS = "source-reviews"
    BUNDLE = "bundle"
    STAGING = "staging"
    REVIEW = "review.json"
    RECIPE = "refinement.json"
    URDF = "refined.urdf"
    BASELINE = "baseline.npz"
    EFFECTIVE = "world.npz"
    VISUAL = "scene.glb"
    COLLISION = "collision.glb"
    MANIFEST = "manifest.json"
    ASSETS = "assets"
    RESULT = "result.json"
    ARCHIVE = "world.tar.gz"
    PROVIDER = "refined_world.py"


# %% Frozen review records
@dataclass(kw_only=True)
class SourceRevision:
    """Source identity and accepted geometry overrides, independent of inspection settings."""

    model_source: str
    """Original URDF path or trusted Python provider identifier."""
    world_sha256: str
    """Imported immutable world snapshot digest."""
    overrides: list[BodyTranslation] = field(default_factory=list)
    """Accepted root-space translations applied after dimension changes."""
    dimension_overrides: list[BodyDimensions] = field(default_factory=list)
    """Accepted cumulative stretch mappings relative to the imported baseline."""

    @classmethod
    def from_project(cls, *, project: RefinementProject) -> SourceRevision:
        """Capture a single coherent accepted model revision."""
        state = project.current_state
        return cls(
            model_source=state.record.model_source,
            world_sha256=state.record.world_sha256,
            overrides=state.overrides,
            dimension_overrides=state.dimension_overrides,
        )

    @property
    def signature(self) -> str:
        """Fingerprint geometry decisions without camera, crop or display settings."""
        return hashlib.sha256(
            json.dumps(
                DataclassJSONSerializer.to_json(instance=self), sort_keys=True
            ).encode()
        ).hexdigest()


@dataclass(kw_only=True)
class SourceReview:
    """A reviewable source application bound to an immutable accepted geometry revision."""

    identifier: str
    """Opaque identity below the project's own review directory."""
    revision: SourceRevision
    """Frozen source and geometry changes displayed before publication."""
    source_sha256: str
    """Digest of the source URDF or the provider entry module."""
    structure: WorldStructure
    """Frame, joint and semantic membership reviewed before publication."""
    changed_bodies: list[str]
    """Qualified bodies changed by accepted overrides, excluding proposals."""


@dataclass(kw_only=True)
class SourceResult:
    """Location and validation status of a completed immutable source-copy bundle."""

    directory: str
    """Project-local output directory containing reloadable source and exports."""
    validated: bool
    """Whether reloaded geometry reproduces the accepted revision."""
    source_kind: str
    """URDF copy or original-provider refinement recipe."""
    body_count: int
    """Bodies retained by the reloaded live world, including geometry-free frames."""
    connection_count: int
    """Preserved kinematic edges, with their original types and degrees of freedom."""
    annotation_count: int
    """Semantic annotations retained from the source provider."""


@dataclass(kw_only=True, frozen=True)
class SurfaceIdentity:
    """Qualified body and geometry intent identifying a whole surface representation."""

    body_name: str
    """Source body identity."""
    role: GeometryRole
    """Appearance or collision representation."""


# %% Kinematic and semantic validation
@dataclass(kw_only=True)
class ConnectionStructure:
    """Joint identity and articulation parameters that source application must preserve."""

    joint_name: str
    """Qualified joint identity retained across reload."""
    parent_name: str
    """Parent frame identity."""
    child_name: str
    """Child frame identity."""
    joint_type: str
    """Concrete SemDT connection class."""
    axis: list[float]
    """Joint axis when the connection has a single active degree of freedom."""
    limits: list[str]
    """Serialized degree-of-freedom names and limits, without runtime object identifiers."""


@dataclass(kw_only=True)
class AnnotationStructure:
    """Semantic identity and referenced frames retained by a provider recipe."""

    annotation_type: str
    """Qualified semantic class name."""
    body_names: list[str]
    """Sorted referenced kinematic frame identities."""


@dataclass(kw_only=True)
class WorldStructure:
    """Source topology and semantics independently of geometry and runtime UUIDs."""

    frame_names: list[str]
    """All body and region names, including geometry-free frames."""
    connections: list[ConnectionStructure]
    """Joint identities, graph edges, types, axes and declared limits."""
    annotations: list[AnnotationStructure]
    """Semantic classes and the frames they reference."""

    @classmethod
    def capture(cls, *, world: World) -> WorldStructure:
        """Record source membership and articulation without changing world state."""
        connections = []
        for connection in sorted(world.connections, key=lambda item: str(item.name)):
            limits = [
                str(dof.name)
                + ":"
                + json.dumps(
                    DataclassJSONSerializer.to_json(instance=dof.limits), sort_keys=True
                )
                for dof in connection.dofs
            ]
            connections.append(
                ConnectionStructure(
                    joint_name=str(connection.name),
                    parent_name=str(connection.parent.name),
                    child_name=str(connection.child.name),
                    joint_type=type(connection).__qualname__,
                    axis=connection.axis.to_np()[:3].tolist()
                    if isinstance(connection, ActiveConnection1DOF)
                    else [],
                    limits=limits,
                )
            )
        annotations = [
            AnnotationStructure(
                annotation_type=type(annotation).__module__
                + "."
                + type(annotation).__qualname__,
                body_names=sorted(
                    str(body.name) for body in annotation.kinematic_structure_entities
                ),
            )
            for annotation in world.semantic_annotations
        ]
        annotations.sort(key=lambda item: (item.annotation_type, item.body_names))
        return cls(
            frame_names=sorted(
                str(body.name) for body in world.kinematic_structure_entities
            ),
            connections=connections,
            annotations=annotations,
        )


# %% Reloadable reviewed world
@dataclass(kw_only=True)
class ReviewedWorldSource(WorldSource):
    """Recreate the original live world with a frozen, verified refinement recipe."""

    directory: Path
    """Bundle directory containing the baseline and refinement recipe."""

    def create_world(self) -> World:
        revision = DataclassJSONSerializer.from_json(
            data=json.loads((self.directory / SourceArtifact.RECIPE).read_text()),
            clazz=SourceRevision,
        )
        review = DataclassJSONSerializer.from_json(
            data=json.loads((self.directory / SourceArtifact.REVIEW).read_text()),
            clazz=SourceReview,
        )
        if (
            revision.signature != review.revision.signature
            or SourceApplication.source_digest(identifier=revision.model_source)
            != review.source_sha256
        ):
            raise InvalidEvidence(
                reason="The provider source or refinement recipe changed after export. Generate a new reviewed copy."
            )
        world = SourceApplication.load_world(identifier=revision.model_source)
        if WorldStructure.capture(world=world) != review.structure:
            raise InvalidEvidence(
                reason="The provider's frame, joint or semantic membership changed after export."
            )
        SourceApplication.check_geometry(
            expected=WorldSnapshot.read(path=self.directory / SourceArtifact.BASELINE),
            actual=WorldSnapshot.from_world(world=world),
        )
        SourceDeformation(world=world, revision=revision).apply()
        SourceApplication.check_geometry(
            expected=WorldSnapshot.read(path=self.directory / SourceArtifact.EFFECTIVE),
            actual=WorldSnapshot.from_world(world=world),
        )
        return world


# %% Live geometry and kinematics
@dataclass(kw_only=True)
class SourceDeformation:
    """Apply selected-body edits while preserving unedited descendant world poses."""

    world: World
    """Fresh source world owned exclusively by this application attempt."""
    revision: SourceRevision
    """Accepted translation and dimension mappings."""
    tolerance: float = 1e-6
    """Metre and matrix tolerance for unchanged geometry directions."""

    def apply(self) -> None:
        """Retain joint types and reject unsupported moving-hinge or inertial resizing."""
        poses = {
            str(body.name): self.world.compute_forward_kinematics_np(
                self.world.root, body
            )
            for body in self.world.kinematic_structure_entities
        }
        translations = {
            item.body_name: np.array(item.translation)
            for item in self.revision.overrides
        }
        dimensions = {
            item.body_name: item for item in self.revision.dimension_overrides
        }
        bodies = {str(body.name): body for body in self.world.bodies}
        if not (translations.keys() | dimensions.keys()) <= bodies.keys():
            raise InvalidEvidence(
                reason="The source no longer contains every changed body."
            )
        for name, translation in translations.items():
            body = bodies[name]
            if np.linalg.norm(translation) > self.tolerance and (
                body == self.world.root
                or not isinstance(
                    body.parent_connection, (FixedConnection, PrismaticConnection)
                )
            ):
                raise InvalidEvidence(
                    reason=f"{name}: source translation currently requires an incoming fixed or prismatic joint; moving hinge frames requires explicit articulation constraints."
                )
        for name in dimensions:
            if bodies[name].inertial is not None:
                raise InvalidEvidence(
                    reason=f"{name}: resizing inertial geometry requires a reviewed mass/inertia policy."
                )
        with self.world.modify_world():
            for connection in self.world.connections:
                parent = str(connection.parent.name)
                child = str(connection.child.name)
                offset = translations.get(child, np.zeros(3)) - translations.get(
                    parent, np.zeros(3)
                )
                if not np.any(offset):
                    continue
                placement = connection.parent_T_connection_expression.to_np().copy()
                placement[:3, 3] += poses[parent][:3, :3].T @ offset
                connection.parent_T_connection_expression = (
                    HomogeneousTransformationMatrix(
                        data=placement, reference_frame=connection.parent
                    )
                )
            for name, override in dimensions.items():
                root_T_body = poses[name]
                for collection in (bodies[name].visual, bodies[name].collision):
                    for index, shape in enumerate(collection.shapes):
                        body_T_shape = shape.origin.to_np()
                        root_T_shape = root_T_body @ body_T_shape
                        root_center = (
                            override.linear @ root_T_shape[:3, 3] + override.offset
                        )
                        placement = body_T_shape.copy()
                        placement[:3, 3] = root_T_body[:3, :3].T @ (
                            root_center - root_T_body[:3, 3]
                        )
                        local_linear = (
                            root_T_shape[:3, :3].T
                            @ override.linear
                            @ root_T_shape[:3, :3]
                        )
                        origin = HomogeneousTransformationMatrix(
                            data=placement, reference_frame=bodies[name]
                        )
                        if isinstance(shape, Box) and np.allclose(
                            local_linear,
                            np.diag(np.diag(local_linear)),
                            atol=self.tolerance,
                            rtol=0,
                        ):
                            size = shape.scale.to_np() * np.diag(local_linear)
                            shape.scale = Scale(x=size[0], y=size[1], z=size[2])
                            shape.origin = origin
                        elif isinstance(shape, Mesh):
                            mesh = shape.mesh.copy()
                            mesh.vertices = mesh.vertices @ local_linear.T
                            replacement = Mesh.from_trimesh(
                                mesh=mesh,
                                origin=origin,
                                file_type=MeshFileFormat.OBJ,
                                finish=shape.finish,
                            )
                            replacement.color = shape.color
                            collection.shapes[index] = replacement
                        else:
                            raise InvalidEvidence(
                                reason=f"{name}: source dimension mapping cannot preserve this primitive type and orientation."
                            )
            for name in dimensions:
                body = bodies[name]
                body.visual = ShapeCollection(
                    shapes=body.visual.shapes, reference_frame=body
                )
                body.collision = ShapeCollection(
                    shapes=body.collision.shapes, reference_frame=body
                )
        self.world.validate()


# %% Source-copy publication
@dataclass(kw_only=True)
class SourceApplication:
    """Create reviewed source copies without overwriting baseline files or project decisions."""

    project: RefinementProject
    """Project whose accepted overrides form the source application revision."""

    def __post_init__(self) -> None:
        """Keep project-local publication paths absolute while preserving original source files."""
        self.project = RefinementProject(directory=self.project.directory.resolve())

    @staticmethod
    def bundle_resolver(*, directory: Path) -> CompositePathResolver:
        """Resolve portable bundle-relative mesh and material resources."""
        return CompositePathResolver(
            resolvers=[
                FileUriResolver(base_directory=str(directory)),
                PackageUriResolver(),
            ]
        )

    @staticmethod
    def load_world(*, identifier: str) -> World:
        """Load a trusted original source with its original qualified body names."""
        path = Path(identifier).resolve()
        if path.suffix == ".urdf" and path.is_file():
            return URDFParser.from_file(
                file_path=str(path),
                path_resolver=SourceApplication.bundle_resolver(directory=path.parent),
            ).create_world()
        module, separator, name = identifier.partition(":")
        if not separator or not module or not name:
            raise InvalidEvidence(
                reason="Source application requires an existing URDF or an importable module:Provider source."
            )
        return vars(import_module(module))[name]().create_world()

    @staticmethod
    def source_digest(*, identifier: str) -> str:
        """Hash the URDF or provider entry module independently of its frozen geometry."""
        path = Path(identifier)
        if not path.is_file():
            module = import_module(identifier.partition(":")[0])
            path = Path(module.__file__)
        return RefinementProject.digest(path=path)

    @staticmethod
    def check_geometry(
        *, expected: WorldSnapshot, actual: WorldSnapshot, tolerance: float = 2e-6
    ) -> None:
        """Require equal labelled surfaces and bidirectional vertex agreement after reload."""
        expected_keys = {
            SurfaceIdentity(body_name=surface.body_name, role=surface.role)
            for surface in expected.meshes
        }
        actual_keys = {
            SurfaceIdentity(body_name=surface.body_name, role=surface.role)
            for surface in actual.meshes
        }
        if expected_keys != actual_keys:
            raise InvalidEvidence(
                reason="The reloaded source has different body/geometry-role identities from the frozen world."
            )
        for identity in expected_keys:
            first = expected.body_mesh(
                body_name=identity.body_name, role=identity.role
            ).mesh
            second = actual.body_mesh(
                body_name=identity.body_name, role=identity.role
            ).mesh
            if (
                len(first.faces) != len(second.faces)
                or not np.isclose(
                    first.area, second.area, atol=tolerance, rtol=tolerance
                )
                or max(
                    cKDTree(first.vertices).query(second.vertices)[0].max(),
                    cKDTree(second.vertices).query(first.vertices)[0].max(),
                )
                > tolerance
            ):
                raise InvalidEvidence(
                    reason=f"{identity.body_name} ({identity.role}): source geometry differs from the reviewed snapshot."
                )

    def prepare(self) -> SourceReview:
        """Freeze accepted edits only after the original source reproduces the baseline."""
        revision = SourceRevision.from_project(project=self.project)
        world = self.load_world(identifier=revision.model_source)
        self.check_geometry(
            expected=self.project.snapshot, actual=WorldSnapshot.from_world(world=world)
        )
        review = SourceReview(
            identifier=uuid4().hex,
            revision=revision,
            source_sha256=self.source_digest(identifier=revision.model_source),
            structure=WorldStructure.capture(world=world),
            changed_bodies=sorted(
                {item.body_name for item in revision.overrides}
                | {item.body_name for item in revision.dimension_overrides}
            ),
        )
        directory = self.project.directory / SourceArtifact.REVIEWS / review.identifier
        directory.mkdir(parents=True)
        (directory / SourceArtifact.REVIEW).write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=review), indent=2)
        )
        return review

    def publish(self, *, identifier: str) -> SourceResult:
        """Publish a new validated source bundle, rejecting stale reviews and existing outputs."""
        if len(identifier) != 32 or any(
            character not in "0123456789abcdef" for character in identifier
        ):
            raise InvalidEvidence(reason="Choose an existing source review identifier.")
        directory = self.project.directory / SourceArtifact.REVIEWS / identifier
        review = DataclassJSONSerializer.from_json(
            data=json.loads((directory / SourceArtifact.REVIEW).read_text()),
            clazz=SourceReview,
        )
        if (
            review.revision.signature
            != SourceRevision.from_project(project=self.project).signature
        ):
            raise InvalidEvidence(
                reason="The geometry changed after source review. Prepare a new review."
            )
        source = Path(review.revision.model_source)
        if (
            self.source_digest(identifier=review.revision.model_source)
            != review.source_sha256
        ):
            raise InvalidEvidence(
                reason="The source changed after review. Prepare a new review."
            )
        output = directory / SourceArtifact.BUNDLE
        if output.exists():
            raise InvalidEvidence(
                reason="This review already has a published output; create a new review."
            )
        world = self.load_world(identifier=review.revision.model_source)
        structure = WorldStructure.capture(world=world)
        if structure != review.structure:
            raise InvalidEvidence(
                reason="Source kinematic or semantic membership changed after review."
            )
        self.check_geometry(
            expected=self.project.snapshot, actual=WorldSnapshot.from_world(world=world)
        )
        SourceDeformation(world=world, revision=review.revision).apply()
        expected = self.project.effective_snapshot
        self.check_geometry(
            expected=expected, actual=WorldSnapshot.from_world(world=world)
        )
        staging = directory / SourceArtifact.STAGING
        staging.mkdir(exist_ok=False)
        shutil.copyfile(
            directory / SourceArtifact.REVIEW, staging / SourceArtifact.REVIEW
        )
        self.project.snapshot.save(path=staging / SourceArtifact.BASELINE)
        expected.save(path=staging / SourceArtifact.EFFECTIVE)
        (staging / SourceArtifact.RECIPE).write_text(
            json.dumps(
                DataclassJSONSerializer.to_json(instance=review.revision), indent=2
            )
        )
        if source.is_file():
            URDFSourceCopy(source=source, world=world, directory=staging).write()
            reloaded = URDFParser.from_file(
                file_path=str(staging / SourceArtifact.URDF),
                path_resolver=self.bundle_resolver(directory=staging),
            ).create_world()
        else:
            reloaded = ReviewedWorldSource(directory=staging).create_world()
        self.check_geometry(
            expected=expected, actual=WorldSnapshot.from_world(world=reloaded)
        )
        if WorldStructure.capture(world=reloaded) != structure:
            raise InvalidEvidence(
                reason="Source reload changed kinematic or semantic membership."
            )
        for role, filename in (
            (GeometryRole.VISUAL, SourceArtifact.VISUAL),
            (GeometryRole.COLLISION, SourceArtifact.COLLISION),
        ):
            scene = trimesh.Scene()
            for surface in WorldSnapshot.from_world(world=reloaded).meshes:
                if surface.role == role:
                    scene.add_geometry(
                        surface.mesh,
                        geom_name=surface.body_name,
                        node_name=surface.body_name,
                        metadata={
                            "body_name": surface.body_name,
                            "geometry_role": str(role),
                        },
                    )
            (staging / filename).write_bytes(scene.export(file_type="glb"))
        result = SourceResult(
            directory=str(output.absolute()),
            validated=True,
            source_kind="urdf" if source.is_file() else "provider-recipe",
            body_count=len(reloaded.bodies),
            connection_count=len(reloaded.connections),
            annotation_count=len(reloaded.semantic_annotations),
        )
        (staging / SourceArtifact.MANIFEST).write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=result), indent=2)
        )
        shutil.copyfile(
            Path(__file__).with_name("refined_world_template.py"),
            staging / SourceArtifact.PROVIDER,
        )
        staging.rename(output)
        archive_path = directory / SourceArtifact.ARCHIVE
        temporary_archive = archive_path.with_suffix(".partial")
        with tarfile.open(temporary_archive, "w:gz") as archive:
            archive.add(output, arcname="refined-world")
        temporary_archive.replace(archive_path)
        (directory / SourceArtifact.RESULT).write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=result), indent=2)
        )
        return result


# %% Portable URDF copy
@dataclass(kw_only=True)
class URDFSourceCopy:
    """Retain source XML identities and joints while writing edited shapes and mesh assets."""

    source: Path
    """Original URDF whose non-geometric metadata is retained."""
    world: World
    """Validated edited source world in its original root frame."""
    directory: Path
    """Unpublished output directory for the refined source and assets."""

    def write(self) -> None:
        """Write independent OBJ assets and update original link and joint parameters."""
        tree = ElementTree.parse(self.source)
        robot = tree.getroot()
        bodies = {body.name.name: body for body in self.world.bodies}
        assets = self.directory / SourceArtifact.ASSETS
        assets.mkdir()
        for link in robot.findall("link"):
            body = bodies[link.attrib["name"]]
            for role, collection in (
                (GeometryRole.VISUAL, body.visual),
                (GeometryRole.COLLISION, body.collision),
            ):
                elements = link.findall(str(role))
                if len(elements) != len(collection.shapes):
                    raise InvalidEvidence(
                        reason="Unsupported source geometry cannot be silently omitted."
                    )
                for element, shape in zip(elements, collection.shapes):
                    origin = element.find("origin")
                    if origin is None:
                        origin = ElementTree.SubElement(element, "origin")
                    placement = shape.origin.to_np()
                    origin.set("xyz", " ".join(map(str, placement[:3, 3])))
                    origin.set(
                        "rpy",
                        " ".join(
                            map(
                                str,
                                trimesh.transformations.euler_from_matrix(placement),
                            )
                        ),
                    )
                    geometry = element.find("geometry")
                    if isinstance(shape, Box):
                        geometry.find("box").set(
                            "size", " ".join(map(str, shape.scale.to_np()))
                        )
                    elif isinstance(shape, Mesh):
                        converted = Mesh.from_trimesh(
                            mesh=shape.mesh.copy(),
                            directory=assets,
                            file_type=MeshFileFormat.OBJ,
                            finish=shape.finish,
                        )
                        mesh = geometry.find("mesh")
                        mesh.set(
                            "filename",
                            str(
                                Path(converted.filename).relative_to(
                                    self.directory.resolve()
                                )
                            ),
                        )
                        mesh.set("scale", "1 1 1")
        joints = {
            connection.name.name: connection for connection in self.world.connections
        }
        for joint in robot.findall("joint"):
            connection = joints[joint.attrib["name"]]
            placement = connection.parent_T_connection_expression.to_np()
            origin = joint.find("origin")
            if origin is None:
                origin = ElementTree.SubElement(joint, "origin")
            origin.set("xyz", " ".join(map(str, placement[:3, 3])))
            origin.set(
                "rpy",
                " ".join(
                    map(str, trimesh.transformations.euler_from_matrix(placement))
                ),
            )
        resolver = SourceApplication.bundle_resolver(directory=self.source.parent)
        for texture in robot.findall(".//texture"):
            original = Path(resolver.resolve(texture.attrib["filename"]))
            destination = assets / (
                hashlib.sha256(original.read_bytes()).hexdigest() + original.suffix
            )
            shutil.copyfile(original, destination)
            texture.set("filename", str(destination.relative_to(self.directory)))
        ElementTree.indent(tree)
        tree.write(
            self.directory / SourceArtifact.URDF, encoding="utf-8", xml_declaration=True
        )
