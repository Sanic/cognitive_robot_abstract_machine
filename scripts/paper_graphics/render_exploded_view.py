#!/usr/bin/env python3
"""
Script to render a world in an exploded view, where all bodies are offset outward
from the geometric center of the world.

Usage:
    python render_exploded_view.py                  # List available worlds
    python render_exploded_view.py <world_name>    # Load and render specific world by name
    python render_exploded_view.py --id <db_id>    # Load and render specific world by database ID
    python render_exploded_view.py --save <path>   # Save rendered image to file
    python render_exploded_view.py --explosion-factor <float>  # Control explosion distance (default: 1.0)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from krrood.ormatic.utils import create_engine

from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO, Base
from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world import World

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


def list_available_worlds(session: Session) -> list[tuple[str, int]]:
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


def calculate_geometric_center(world: World) -> np.ndarray:
    """
    Calculate the geometric center of all bodies in the world.
    Returns the center point as a 3D numpy array.
    """
    positions = []

    for body in world.bodies:
        # Get body position in world frame
        body_transform = world.compute_forward_kinematics_np(world.root, body)
        # Extract translation (last column, first 3 rows)
        position = body_transform[:3, 3]
        positions.append(position)

    if not positions:
        return np.array([0.0, 0.0, 0.0])

    return np.mean(positions, axis=0)


def create_exploded_world(world: World, explosion_factor: float = 1.0) -> World:
    """
    Create a copy of the world with all bodies offset outward from the geometric center.

    Args:
        world: The original world
        explosion_factor: Multiplier for the explosion distance (1.0 = normal, 2.0 = double distance)

    Returns:
        A new World with bodies exploded outward
    """
    # Calculate geometric center
    center = calculate_geometric_center(world)
    print(f"Geometric center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")

    # Create a copy of the world to modify
    import copy

    exploded_world = copy.deepcopy(world)

    # Calculate average distance from center to determine explosion distance
    distances = []
    for body in world.bodies:
        body_transform = world.compute_forward_kinematics_np(world.root, body)
        position = body_transform[:3, 3]
        distance = np.linalg.norm(position - center)
        distances.append(distance)

    avg_distance = np.mean(distances) if distances else 1.0
    explosion_distance = avg_distance * explosion_factor
    print(f"Average distance from center: {avg_distance:.3f}")
    print(f"Explosion distance: {explosion_distance:.3f}")

    # Modify connections to offset bodies
    with exploded_world.modify_world():
        for connection in exploded_world.connections:
            # Only modify connections from root to direct children
            if connection.parent == exploded_world.root:
                # Get original body position in world frame
                original_transform = world.compute_forward_kinematics_np(
                    world.root, connection.child
                )
                original_position = original_transform[:3, 3]

                # Calculate direction from center to body
                direction = original_position - center
                direction_norm = np.linalg.norm(direction)

                if direction_norm > 1e-6:  # Avoid division by zero
                    # Normalize direction
                    direction_unit = direction / direction_norm

                    # Calculate offset vector (how much to move outward)
                    offset_vector = direction_unit * explosion_distance

                    # Get original connection transform (do not convert to numpy)
                    original_connection_transform = connection.origin

                    # Use to_position() to get the original translation as a Point3
                    original_connection_position = (
                        original_connection_transform.to_position()
                    )

                    # offset_vector is still numpy; convert to list and unpack for Point3 addition
                    offset_expr = type(original_connection_position)(
                        *(
                            original_connection_position[i] + offset_vector[i]
                            for i in range(3)
                        ),
                        reference_frame=original_connection_position.reference_frame,
                    )

                    # Create new transform by copying and updating translation; preserve rotation
                    new_transform = TransformationMatrix(
                        data=original_connection_transform,
                        reference_frame=original_connection_transform.reference_frame,
                        child_frame=original_connection_transform.child_frame,
                    )
                    new_transform.x = offset_expr[0]
                    new_transform.y = offset_expr[1]
                    new_transform.z = offset_expr[2]

                    # Update connection origin
                    connection.origin = new_transform

                    # Calculate new world position for display
                    new_world_transform = exploded_world.compute_forward_kinematics_np(
                        exploded_world.root, connection.child
                    )
                    new_world_position = new_world_transform[:3, 3]

                    print(
                        f"  Exploded {connection.child.name.name}: "
                        f"({original_position[0]:.3f}, {original_position[1]:.3f}, {original_position[2]:.3f}) -> "
                        f"({new_world_position[0]:.3f}, {new_world_position[1]:.3f}, {new_world_position[2]:.3f})"
                    )

    return exploded_world


def render_exploded_world(
    world: World,
    save_path: Optional[Path] = None,
    show: bool = True,
    camera_index: int = 0,
    explosion_factor: float = 1.0,
) -> bytes:
    """
    Render the world in an exploded view.

    Args:
        world: The world to render
        save_path: Optional path to save the rendered image
        show: Whether to show the interactive viewer
        camera_index: Index of predefined camera pose (0-3)
        explosion_factor: Multiplier for explosion distance

    Returns:
        PNG image data as bytes
    """
    # Create exploded version of the world
    print("Creating exploded view...")
    exploded_world = create_exploded_world(world, explosion_factor=explosion_factor)

    # Use WarsawWorldLoader for camera setup
    world_loader = WarsawWorldLoader.from_world(exploded_world)

    # Create RayTracer scene
    rt = RayTracer(exploded_world)
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
            print("\nUsage:")
            print("  python render_exploded_view.py <world_name>  # Load by name")
            print(
                "  python render_exploded_view.py --id <db_id>  # Load by database ID"
            )
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

        # Render the exploded world
        if not args.no_render:
            render_exploded_world(
                world,
                save_path=args.save,
                camera_index=args.camera,
                explosion_factor=args.explosion_factor,
            )

    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and render a world in an exploded view."
    )
    parser.add_argument(
        "world_name",
        nargs="?",
        help="Name of the world to load. If not provided, lists available worlds.",
    )
    parser.add_argument(
        "--id",
        type=int,
        metavar="DB_ID",
        help="Load world by database ID instead of name. Use this option to specify a world by its database ID (shown in brackets when listing worlds).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Path to save the rendered image.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering, only load the world.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Predefined camera pose index (0-3, default: 0).",
    )
    parser.add_argument(
        "--explosion-factor",
        type=float,
        default=1.0,
        help="Multiplier for explosion distance. 1.0 = normal explosion, 2.0 = double distance, etc. (default: 1.0).",
    )

    main(parser.parse_args())
