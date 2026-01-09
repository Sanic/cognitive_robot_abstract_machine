#!/usr/bin/env python3
"""
Script to load a world and render it with each object painted in a different color.

Usage:
    python render_colored_objects.py                  # List available worlds
    python render_colored_objects.py <world_name>    # Load and render specific world by name
    python render_colored_objects.py --id <db_id>    # Load and render specific world by database ID
    python render_colored_objects.py --save <path>   # Save rendered image to file
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from sqlalchemy import select
from sqlalchemy.orm import Session
from krrood.ormatic.utils import create_engine

# Import ORM interface - this brings in all DAOs including WorldMappingDAO
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO, Base
from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
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


def generate_distinct_colors(n: int) -> list[np.ndarray]:
    """
    Generate n visually distinct colors using HSV color space with golden ratio spacing.
    Ensures each color is unique and maximally separated from others.
    """
    import colorsys

    if n == 0:
        return []

    if n == 1:
        # Single color - use a bright, distinct color
        r, g, b = colorsys.hsv_to_rgb(0.0, 0.9, 0.9)
        return [
            np.array([int(r * 255), int(g * 255), int(b * 255), 255], dtype=np.uint8)
        ]

    colors = []
    # Use golden ratio for optimal color distribution (ensures colors don't cluster)
    golden_ratio = 0.618033988749895

    # Track used RGB values to ensure uniqueness
    used_rgb_values = set()

    for i in range(n):
        # Use golden ratio to space hues evenly around the color wheel
        hue = (i * golden_ratio) % 1.0

        # Vary saturation and value to increase distinction
        # Cycle through different saturation/value combinations
        sat_val_combos = [
            (0.9, 0.9),
            (0.85, 0.95),
            (0.95, 0.85),
            (0.8, 0.9),
            (0.9, 0.8),
        ]
        saturation, value = sat_val_combos[i % len(sat_val_combos)]

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

        # Convert to uint8 RGB values
        r_int = int(r * 255)
        g_int = int(g * 255)
        b_int = int(b * 255)

        # Ensure uniqueness by checking RGB tuple
        rgb_tuple = (r_int, g_int, b_int)
        attempts = 0
        max_attempts = 100

        while rgb_tuple in used_rgb_values and attempts < max_attempts:
            # Shift hue slightly to get a different color
            hue = (hue + 0.05) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            r_int = int(r * 255)
            g_int = int(g * 255)
            b_int = int(b * 255)
            rgb_tuple = (r_int, g_int, b_int)
            attempts += 1

        # Add to used set and colors list
        used_rgb_values.add(rgb_tuple)
        color = np.array([r_int, g_int, b_int, 255], dtype=np.uint8)
        colors.append(color)

    return colors


def colorize_objects(scene: trimesh.Scene, world: World) -> dict[str, np.ndarray]:
    """
    Colorize each body's collision meshes with a distinct color.

    Args:
        scene: The trimesh scene containing the meshes
        world: The world containing bodies

    Returns:
        Dictionary mapping body names to their colors
    """
    # Get all bodies that have collision geometry
    bodies_with_collision = [body for body in world.bodies if body.collision]

    if not bodies_with_collision:
        print("Warning: No bodies with collision geometry found.")
        return {}

    # Generate distinct colors for each body
    colors = generate_distinct_colors(len(bodies_with_collision))
    body_to_color = {}

    # Assign colors to each body's collision meshes
    for i, body in enumerate(bodies_with_collision):
        color = colors[i]
        body_to_color[body.name.name] = color

        # Find all collision meshes for this body in the scene
        for j in range(len(body.collision)):
            node_name = body.name.name + f"_collision_{j}"

            # Get the geometry from the scene
            if node_name in scene.graph.nodes_geometry:
                # scene.graph[node_name] returns (transform, geometry_name)
                geometry_name = scene.graph[node_name][1]
                geometry = scene.geometry[geometry_name]

                # Set face colors for the mesh
                if hasattr(geometry, "visual") and hasattr(
                    geometry.visual, "face_colors"
                ):
                    # Set all faces to the same color
                    num_faces = len(geometry.faces)
                    geometry.visual.face_colors = np.tile(color, (num_faces, 1))
                    print(
                        f"Colored {node_name} with RGB({color[0]}, {color[1]}, {color[2]})"
                    )

    return body_to_color


def print_color_legend(body_to_color: dict[str, np.ndarray]) -> None:
    """Print a legend mapping colors to body names."""
    if not body_to_color:
        return
    print("\n" + "=" * 60)
    print("OBJECT COLOR LEGEND")
    print("=" * 60)
    for body_name, color in body_to_color.items():
        r, g, b = color[:3]
        # Print with ANSI color codes for terminal visualization
        print(f"  \033[38;2;{r};{g};{b}m●\033[0m  {body_name} (RGB: {r}, {g}, {b})")
    print("=" * 60 + "\n")


def render_world(
    world: World,
    save_path: Optional[Path] = None,
    show: bool = True,
    camera_index: int = 0,
) -> bytes:
    """
    Render the world with each object painted in a different color.

    Args:
        world: The world to render
        save_path: Optional path to save the rendered image
        show: Whether to show the interactive viewer
        camera_index: Index of predefined camera pose (0-3)

    Returns:
        PNG image data as bytes
    """
    # Use WarsawWorldLoader for camera setup
    world_loader = WarsawWorldLoader.from_world(world)

    # Create RayTracer scene
    rt = RayTracer(world)
    rt.update_scene()
    scene = rt.scene

    # Colorize all objects with distinct colors
    print("Coloring objects...")
    body_to_color = colorize_objects(scene, world)
    print_color_legend(body_to_color)

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
            print("  python render_colored_objects.py <world_name>  # Load by name")
            print(
                "  python render_colored_objects.py --id <db_id>  # Load by database ID"
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

        # Render the world
        if not args.no_render:
            render_world(
                world,
                save_path=args.save,
                camera_index=args.camera,
            )

    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and render a world with each object painted in a different color."
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

    main(parser.parse_args())
