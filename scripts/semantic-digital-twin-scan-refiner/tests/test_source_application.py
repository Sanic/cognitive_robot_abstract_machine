"""
Source copies must reload as the exact reviewed geometry without replacing baselines.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import numpy as np
import trimesh
from scan_refiner.records import GeometryRole
import pytest
import shutil
import os
import threading
import json
import urllib.request
from urllib.error import HTTPError
from http import HTTPStatus
from typing_extensions import Iterator
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.web_service import InspectionService, InspectionServer
from scan_refiner.source_application import SourceReview, SourceResult
from scan_refiner.project import RefinementProject
from scan_refiner.world import WorldSnapshot
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.source_application import (
    SourceApplication,
    SourceArtifact,
    ReviewedWorldSource,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from scan_refiner.refinement import BodyTranslation


# %% Real source evidence
@pytest.fixture
def source_project(tmp_path: Path) -> RefinementProject:
    """
    Fixed children and a hinged door exercise source topology preservation.
    """
    source = Path(__file__).parent / "dataset/source_world.urdf"
    world = URDFParser.from_file(file_path=str(source)).create_world()
    snapshot = WorldSnapshot.from_world(world=world)
    scan = tmp_path / "scan.npz"
    np.savez(
        scan,
        points=snapshot.meshes[0].mesh.vertices,
        colors=np.full(snapshot.meshes[0].mesh.vertices.shape, 128, dtype=np.uint8),
    )
    return RefinementProject.create(
        directory=tmp_path / "project",
        scan_path=scan,
        snapshot=snapshot,
        model_source=str(source.absolute()),
    )


# %% Reviewed publication
def test_source_copy_reloads_a_group_move_without_double_moving_the_child(
    source_project: RefinementProject,
) -> None:
    """
    A fixed parent's and child's requested deltas survive native export once.
    """
    source_project.apply_translations(
        edits=[
            BodyTranslation(body_name=name, translation=[0.0, 0.012, 0.0])
            for name in ("test/box", "test/handle")
        ],
        label="Translate the fixed pair",
    )
    service = SourceApplication(project=source_project)
    review = service.prepare()
    result = service.publish(identifier=review.identifier)
    assert result.validated
    assert set(review.changed_bodies) == {"test/box", "test/handle"}


def test_source_application_writes_reloaded_copy_and_preserves_children(
    source_project: RefinementProject,
) -> None:
    """
    A selected body moves while attached geometry and the original files stay fixed.
    """
    source = Path(source_project.record.model_source)
    original = source.read_bytes()
    source_project.apply_translation(
        body_name="test/box", translation=[0.02, -0.01, 0.03], source_job="reviewed"
    )
    service = SourceApplication(project=source_project)
    review = service.prepare()
    result = service.publish(identifier=review.identifier)
    copied = Path(result.directory) / SourceArtifact.URDF
    world = URDFParser.from_file(
        file_path=str(copied),
        path_resolver=service.bundle_resolver(directory=copied.parent),
    ).create_world()
    actual = WorldSnapshot.from_world(world=world)
    service.check_geometry(expected=source_project.effective_snapshot, actual=actual)
    assert source.read_bytes() == original
    assert result.validated
    assert len(world.connections) == 3
    assert (copied.parent / SourceArtifact.VISUAL).is_file()
    assert (copied.parent / SourceArtifact.COLLISION).is_file()
    source_project.undo()
    assert source_project.current_state.overrides == []


def test_source_review_is_stale_after_geometry_undo(
    source_project: RefinementProject,
) -> None:
    """
    An earlier review cannot publish after its accepted geometry has changed.
    """
    source_project.apply_translation(
        body_name="test/box", translation=[0.02, 0, 0], source_job="reviewed"
    )
    service = SourceApplication(project=source_project)
    review = service.prepare()
    source_project.undo()
    with pytest.raises(InvalidEvidence):
        service.publish(identifier=review.identifier)


def test_changed_source_is_rejected_before_export(
    source_project: RefinementProject, tmp_path: Path
) -> None:
    """
    A source that no longer reproduces the imported baseline must not absorb overrides.
    """
    path = tmp_path / "changed.urdf"
    path.write_text(
        Path(source_project.record.model_source)
        .read_text()
        .replace("1 .6 .4", "2 .6 .4")
    )
    state = source_project.current_state
    state.record.model_source = str(path)
    source_project.publish(state=state, label="source changed")
    with pytest.raises(InvalidEvidence):
        SourceApplication(project=source_project).prepare()


# %% Dimensions and articulation
def test_resized_source_primitives_reload_with_original_role_difference(
    source_project: RefinementProject,
) -> None:
    """
    A verified rotated box keeps its directions and a deliberately smaller collision.
    """
    rotation = trimesh.transformations.rotation_matrix(0.3, [0, 0, 1])[:3, :3]
    matrix = rotation @ np.diag([1.04, 1, 1]) @ rotation.T
    center = source_project.snapshot.body_mesh(
        body_name="test/box", role=GeometryRole.VISUAL
    ).mesh.bounds.mean(axis=0)
    source_project.apply_dimensions(
        body_name="test/box",
        scale=[1, 1, 1],
        offset=(center - matrix @ center).tolist(),
        source_job="reviewed",
        matrix=matrix.tolist(),
    )
    service = SourceApplication(project=source_project)
    review = service.prepare()
    result = service.publish(identifier=review.identifier)
    assert result.validated
    reloaded = URDFParser.from_file(
        file_path=str(Path(result.directory) / SourceArtifact.URDF),
        path_resolver=service.bundle_resolver(directory=Path(result.directory)),
    ).create_world()
    service.check_geometry(
        expected=source_project.effective_snapshot,
        actual=WorldSnapshot.from_world(world=reloaded),
    )


def test_hinge_translation_requires_articulation_policy(
    source_project: RefinementProject,
) -> None:
    """
    The source writer must not silently move a rotating door's hinge frame.
    """
    source_project.apply_translation(
        body_name="test/door", translation=[0.02, 0, 0], source_job="reviewed"
    )
    service = SourceApplication(project=source_project)
    review = service.prepare()
    with pytest.raises(InvalidEvidence):
        service.publish(identifier=review.identifier)
    assert not (
        source_project.directory / SourceArtifact.REVIEWS / review.identifier / "bundle"
    ).exists()


# %% Providers and portable mesh assets
def test_provider_recipe_keeps_semantics_and_reloads_after_undo(
    source_project: RefinementProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Exported providers retain semantics and remain independent of later project Undo.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "dataset"))
    state = source_project.current_state
    state.record.model_source = "source_provider:FurnitureWorld"
    source_project.publish(state=state, label="provider source")
    source_project.apply_translation(
        body_name="test/box", translation=[0.02, 0, 0], source_job="reviewed"
    )
    service = SourceApplication(project=source_project)
    review = service.prepare()
    result = service.publish(identifier=review.identifier)
    expected = source_project.effective_snapshot
    source_project.undo()
    world = ReviewedWorldSource(directory=Path(result.directory)).create_world()
    service.check_geometry(
        expected=expected, actual=WorldSnapshot.from_world(world=world)
    )
    assert len(world.semantic_annotations) == 1
    assert world.semantic_annotations[0].description == "Fixture cabinet"
    assert result.source_kind == "provider-recipe"


@pytest.mark.parametrize("relative_project", [False, True])
def test_mesh_source_bundle_reloads_after_relocation(
    source_project: RefinementProject, tmp_path: Path, relative_project: bool
) -> None:
    """
    OBJ resources are relative and independent of both the original URDF and output
    path.
    """
    mesh = trimesh.creation.box(extents=[1, 0.6, 0.4])
    mesh.export(tmp_path / "body.obj")
    source = tmp_path / "mesh.urdf"
    source.write_text(
        Path(source_project.record.model_source)
        .read_text()
        .replace('<box size="1 .6 .4"/>', '<mesh filename="body.obj"/>')
    )
    state = source_project.current_state
    state.record.model_source = str(source)
    source_project.publish(state=state, label="mesh source")
    source_project.apply_translation(
        body_name="test/box", translation=[0.02, 0, 0], source_job="reviewed"
    )
    service = (
        SourceApplication(
            project=RefinementProject(
                directory=Path(os.path.relpath(source_project.directory))
            )
        )
        if relative_project
        else SourceApplication(project=source_project)
    )
    review = service.prepare()
    result = service.publish(identifier=review.identifier)
    relocated = tmp_path / "relocated"
    shutil.copytree(result.directory, relocated)
    world = URDFParser.from_file(
        file_path=str(relocated / SourceArtifact.URDF),
        path_resolver=service.bundle_resolver(directory=relocated),
    ).create_world()
    service.check_geometry(
        expected=source_project.effective_snapshot,
        actual=WorldSnapshot.from_world(world=world),
    )
    assert list((relocated / SourceArtifact.ASSETS).rglob("*.obj"))


def test_source_digest_change_after_review_blocks_publication(
    source_project: RefinementProject, tmp_path: Path
) -> None:
    """
    Even a non-geometric source edit requires a fresh review of its content.
    """
    source = tmp_path / "source.urdf"
    shutil.copyfile(source_project.record.model_source, source)
    state = source_project.current_state
    state.record.model_source = str(source)
    source_project.publish(state=state, label="source copy")
    service = SourceApplication(project=source_project)
    review = service.prepare()
    source.write_text(source.read_text() + "\n<!-- newer source -->\n")
    with pytest.raises(InvalidEvidence):
        service.publish(identifier=review.identifier)


def test_source_publication_does_not_overwrite_an_existing_bundle(
    source_project: RefinementProject,
) -> None:
    """
    Completed source-copy outputs remain immutable and require a new review for another
    export.
    """
    service = SourceApplication(project=source_project)
    review = service.prepare()
    result = service.publish(identifier=review.identifier)
    digest = source_project.digest(path=Path(result.directory) / SourceArtifact.URDF)
    with pytest.raises(InvalidEvidence):
        service.publish(identifier=review.identifier)
    assert (
        source_project.digest(path=Path(result.directory) / SourceArtifact.URDF)
        == digest
    )


# %% HTTP source workflow
@pytest.fixture
def source_server(source_project: RefinementProject) -> Iterator[InspectionServer]:
    """
    Serve source-copy actions with the same ownership lock as project history.
    """
    with InspectionService(project=source_project).create_server(port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=10)


def test_http_source_review_and_publication_preserve_project_revision(
    source_project: RefinementProject, source_server: InspectionServer
) -> None:
    """
    A local user can review, publish and download a copy without rebasing the project.
    """
    before = source_project.current_state
    url = f"http://127.0.0.1:{source_server.server_port}"
    with urllib.request.urlopen(
        urllib.request.Request(url + "/api/source/review", data=b""), timeout=30
    ) as response:
        review = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=SourceReview
        )
    with urllib.request.urlopen(
        urllib.request.Request(
            url + "/api/source/publish", data=review.identifier.encode()
        ),
        timeout=30,
    ) as response:
        result = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=SourceResult
        )
    assert result.validated
    assert source_project.current_state == before
    with urllib.request.urlopen(
        url + "/source-reviews/" + review.identifier + "/world.tar.gz"
    ) as response:
        assert response.status == HTTPStatus.OK
        assert response.headers["Content-Type"] == "application/gzip"


def test_http_source_copy_rejects_foreign_origin(
    source_server: InspectionServer,
) -> None:
    """
    A foreign page cannot trigger source reloads or filesystem publication.
    """
    url = f"http://127.0.0.1:{source_server.server_port}/api/source/review"
    request = urllib.request.Request(
        url, data=b"", headers={"Origin": "https://example.invalid"}
    )
    with pytest.raises(HTTPError) as failure:
        urllib.request.urlopen(request)
    assert failure.value.code == HTTPStatus.FORBIDDEN


# %% Provider provenance after publication
def test_exported_provider_rejects_later_semantic_source_edits(
    source_project: RefinementProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An unchanged shape must not conceal semantic source edits made after export.
    """
    dataset = Path(__file__).parent / "dataset"
    module_path = tmp_path / "source_provider_copy.py"
    shutil.copyfile(dataset / "source_provider.py", module_path)
    shutil.copyfile(dataset / "source_world.urdf", tmp_path / "source_world.urdf")
    monkeypatch.syspath_prepend(str(tmp_path))
    state = source_project.current_state
    state.record.model_source = "source_provider_copy:FurnitureWorld"
    source_project.publish(state=state, label="provider source")
    service = SourceApplication(project=source_project)
    review = service.prepare()
    result = service.publish(identifier=review.identifier)
    module_path.write_text(
        module_path.read_text().replace("Fixture cabinet", "Changed description")
    )
    with pytest.raises(InvalidEvidence):
        ReviewedWorldSource(directory=Path(result.directory)).create_world()
