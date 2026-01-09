#!/usr/bin/env python3
"""
Script to load a world and its semantic annotations from the database via Ormatic,
and render it with semantic annotation information displayed.

Usage:
    python load_and_render_scene.py                  # List available worlds
    python load_and_render_scene.py <world_name>    # Load and render specific world
    python load_and_render_scene.py --save <path>   # Save rendered image to file
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh
from sqlalchemy import select
from sqlalchemy.orm import Session
from krrood.ormatic.utils import create_engine

# Import ORM interface - this brings in all DAOs including WorldMappingDAO
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO, Base
from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# Database connection settings from environment
DB_NAME = os.getenv("PGDATABASE")
DB_USER = os.getenv("PGUSER")
DB_PASSWORD = os.getenv("PGPASSWORD")
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5432")


def get_connection_string() -> str:
    """Build database connection string from environment variables."""
    if not all([DB_NAME, DB_USER, DB_PASSWORD]):
        raise EnvironmentError(
            "Database credentials not set. Please set PGDATABASE, PGUSER, and PGPASSWORD environment variables."
        )
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def list_available_worlds(session: Session) -> list[str]:
    """List all available world names in the database."""
    query = select(WorldMappingDAO.name, WorldMappingDAO.database_id)
    results = session.execute(query).fetchall()
    return [(r[0], r[1]) for r in results]


def load_world_by_name(session: Session, name: str) -> Optional[World]:
    """Load a world from the database by its name."""
    query = select(WorldMappingDAO).where(WorldMappingDAO.name == name)
    result = session.scalars(query).first()
    if result is None:
        return None
    return result.from_dao()


def load_world_by_id(session: Session, db_id: int) -> Optional[World]:
    """Load a world from the database by its database ID."""
    query = select(WorldMappingDAO).where(WorldMappingDAO.database_id == db_id)
    result = session.scalars(query).first()
    if result is None:
        return None
    return result.from_dao()


def print_semantic_annotations(world: World) -> None:
    """Print all semantic annotations in the world with their associated bodies."""
    print("\n" + "=" * 60)
    print("SEMANTIC ANNOTATIONS")
    print("=" * 60)

    if not world.semantic_annotations:
        print("No semantic annotations found in this world.")
        return

    for i, annotation in enumerate(world.semantic_annotations, 1):
        print(f"\n{i}. {annotation.__class__.__name__}")
        print(f"   Name: {annotation.name}")

        # Get associated bodies
        bodies = list(annotation.bodies)
        if bodies:
            print(f"   Bodies ({len(bodies)}):")
            for body in bodies:
                print(f"      - {body.name}")
        else:
            print("   Bodies: None")

    print("\n" + "=" * 60)
    print(f"Total: {len(world.semantic_annotations)} semantic annotations")
    print("=" * 60 + "\n")


def print_bodies_with_annotations(world: World) -> None:
    """Print each body and its associated semantic annotations."""
    print("\n" + "=" * 60)
    print("BODIES WITH SEMANTIC ANNOTATIONS")
    print("=" * 60)

    for body in world.bodies:
        annotations = body._semantic_annotations
        if annotations:
            print(f"\n{body.name}:")
            for ann in annotations:
                print(f"   -> {ann.__class__.__name__}: {ann.name}")

    print("=" * 60 + "\n")


def get_body_semantic_labels(world: World) -> Dict[Body, List[str]]:
    """
    Build a mapping from bodies to their semantic annotation class names.
    """
    body_labels: Dict[Body, List[str]] = {}
    for annotation in world.semantic_annotations:
        for body in annotation.bodies:
            if body not in body_labels:
                body_labels[body] = []
            body_labels[body].append(annotation.__class__.__name__)
    return body_labels


def compute_body_centroid(world: World, body: Body) -> np.ndarray:
    """
    Compute the world-frame centroid of a body's collision geometry.
    """
    if not body.collision:
        # Fallback: use the body's origin
        fk = world.compute_forward_kinematics_np(world.root, body)
        return fk[:3, 3]

    # Get all collision mesh vertices in world frame
    all_vertices = []
    fk = world.compute_forward_kinematics_np(world.root, body)
    for collision in body.collision:
        mesh = collision.mesh
        local_transform = collision.origin.to_np()
        world_transform = fk @ local_transform
        # Transform vertices to world frame
        vertices_homogeneous = np.hstack(
            [mesh.vertices, np.ones((len(mesh.vertices), 1))]
        )
        world_vertices = (world_transform @ vertices_homogeneous.T).T[:, :3]
        all_vertices.append(world_vertices)

    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        return all_vertices.mean(axis=0)
    else:
        return fk[:3, 3]


def generate_distinct_colors(n: int) -> List[np.ndarray]:
    """Generate n visually distinct colors using HSV color space."""
    import colorsys

    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        # Convert HSV to RGB (saturation=0.9, value=0.9 for vivid colors)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(np.array([int(r * 255), int(g * 255), int(b * 255), 255]))
    return colors


def add_semantic_labels_to_scene(
    scene: trimesh.Scene, world: World, marker_radius: float = 0.03
) -> Dict[str, np.ndarray]:
    """
    Add colored sphere markers for semantic annotations to the scene.
    Each unique semantic annotation class gets a distinct color.

    Args:
        scene: The trimesh scene to add markers to
        world: The world containing bodies and semantic annotations
        marker_radius: Radius of the marker spheres in world units

    Returns:
        Dictionary mapping annotation class names to their colors (for legend)
    """
    body_labels = get_body_semantic_labels(world)

    # Collect all unique annotation class names
    all_classes = set()
    for labels in body_labels.values():
        all_classes.update(labels)
    all_classes = sorted(all_classes)

    if not all_classes:
        return {}

    # Generate distinct colors for each class
    colors = generate_distinct_colors(len(all_classes))
    class_to_color = {cls: colors[i] for i, cls in enumerate(all_classes)}

    # Add markers for each body
    for body, labels in body_labels.items():
        centroid = compute_body_centroid(world, body)

        # Add a marker for each annotation on this body (offset vertically to stack)
        for i, label in enumerate(labels):
            color = class_to_color[label]

            # Create sphere marker
            marker = trimesh.creation.icosphere(subdivisions=2, radius=marker_radius)
            marker.visual.face_colors = color

            # Position above the body centroid, stacked if multiple annotations
            offset = np.array([0, 0, marker_radius * 2.5 * (i + 1)])
            transform = np.eye(4)
            transform[:3, 3] = centroid + offset

            scene.add_geometry(
                marker,
                node_name=f"marker_{body.name.name}_{label}",
                transform=transform,
            )

    return class_to_color


def print_color_legend(class_to_color: Dict[str, np.ndarray]) -> None:
    """Print a legend mapping colors to semantic annotation classes."""
    if not class_to_color:
        return
    print("\n" + "=" * 60)
    print("SEMANTIC ANNOTATION COLOR LEGEND")
    print("=" * 60)
    for class_name, color in class_to_color.items():
        r, g, b = color[:3]
        # Print with ANSI color codes for terminal visualization
        print(f"  \033[38;2;{r};{g};{b}m●\033[0m  {class_name} (RGB: {r}, {g}, {b})")
    print("=" * 60 + "\n")


def render_world(
    world: World,
    save_path: Optional[Path] = None,
    show: bool = True,
    show_labels: bool = True,
    marker_radius: float = 0.03,
    camera_index: int = 0,
) -> bytes:
    """
    Render the world using WarsawWorldLoader with mesh textures and semantic annotation markers.

    Args:
        world: The world to render
        save_path: Optional path to save the rendered image
        show: Whether to show the interactive viewer
        show_labels: Whether to render semantic annotation markers in 3D
        marker_radius: Radius of marker spheres in world units
        camera_index: Index of predefined camera pose (0-3)

    Returns:
        PNG image data as bytes
    """
    # Use WarsawWorldLoader for textured rendering with proper camera setup
    world_loader = WarsawWorldLoader.from_world(world)

    # Create RayTracer scene with textures
    rt = RayTracer(world)
    rt.update_scene()
    scene = rt.scene

    # Set up camera from predefined poses
    camera_poses = world_loader._predefined_camera_transforms
    if 0 <= camera_index < len(camera_poses):
        camera_pose = camera_poses[camera_index]
    else:
        camera_pose = camera_poses[0]

    scene.camera.fov = world_loader._camera_field_of_view
    scene.graph[scene.camera.name] = camera_pose

    # Add semantic label markers to the scene
    if show_labels:
        print("Adding semantic annotation markers to scene...")
        class_to_color = add_semantic_labels_to_scene(
            scene, world, marker_radius=marker_radius
        )
        print_color_legend(class_to_color)

    # Save image if path provided
    if save_path:
        png_data = scene.save_image(resolution=(1920, 1080), visible=True)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to: {save_path}")

    # Show interactive viewer
    if show:
        print("Opening interactive viewer...")
        scene.show()

    return scene.save_image(resolution=(1920, 1080), visible=True) if save_path else b""


def main(args):
    # Connect to database
    try:
        connection_string = get_connection_string()
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    engine = create_engine(connection_string)
    session = Session(engine)

    try:
        # If no world specified, list available worlds
        if args.world_name is None and args.id is None:
            print("Available worlds in database:")
            print("-" * 40)
            worlds = list_available_worlds(session)
            if not worlds:
                print("No worlds found in database.")
            else:
                for name, db_id in worlds:
                    print(f"  [{db_id}] {name}")
            print("-" * 40)
            print(f"Total: {len(worlds)} worlds")
            print("\nUsage: python load_and_render_scene.py <world_name>")
            return

        # Load the specified world
        if args.id is not None:
            print(f"Loading world with ID: {args.id}...")
            world = load_world_by_id(session, args.id)
            if world is None:
                print(f"Error: No world found with ID '{args.id}'", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Loading world: {args.world_name}...")
            world = load_world_by_name(session, args.world_name)
            if world is None:
                print(
                    f"Error: No world found with name '{args.world_name}'",
                    file=sys.stderr,
                )
                sys.exit(1)

        print(f"World loaded: {world.name}")
        print(f"  Bodies: {len(list(world.bodies))}")
        print(f"  Connections: {len(world.connections)}")
        print(f"  Semantic annotations: {len(world.semantic_annotations)}")

        # Print semantic annotations
        print_semantic_annotations(world)

        if args.bodies:
            print_bodies_with_annotations(world)

        # Render the world
        if not args.no_render:
            render_world(
                world,
                save_path=args.save,
                show_labels=not args.no_labels,
                marker_radius=args.marker_radius,
                camera_index=args.camera,
            )

    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and render a world with semantic annotations from the database."
    )
    parser.add_argument(
        "world_name",
        nargs="?",
        help="Name of the world to load. If not provided, lists available worlds.",
    )
    parser.add_argument(
        "--id",
        type=int,
        help="Load world by database ID instead of name.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Path to save the rendered image.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering, only print semantic annotations.",
    )
    parser.add_argument(
        "--bodies",
        action="store_true",
        help="Also print bodies with their annotations.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't render semantic annotation markers in the 3D scene.",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=0.03,
        help="Radius of marker spheres in world units (default: 0.03).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Predefined camera pose index (0-3, default: 0).",
    )

    main(parser.parse_args())
