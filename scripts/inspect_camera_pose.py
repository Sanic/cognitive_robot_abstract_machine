#!/usr/bin/env python3
"""
Auxiliary script to load a world and render it with original textures.
Allows interactive camera rotation, and pressing 'P' prints the current camera transform.

Usage:
    python inspect_camera_pose.py <obj_dir>              # Load from obj directory
    python inspect_camera_pose.py --id <db_id>          # Load from database by ID
    python inspect_camera_pose.py --name <world_name>   # Load from database by name
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

from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO
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


def format_transform_matrix(matrix: np.ndarray) -> str:
    """Format a 4x4 transformation matrix as a Python code string."""
    lines = ["np.array(["]
    for i in range(4):
        row = matrix[i]
        lines.append(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}],")
    lines.append("])")
    return "\n".join(lines)


def format_transform_xyz_rpy(matrix: np.ndarray) -> str:
    """Try to extract xyz_rpy from matrix and format it."""
    # Extract translation
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]

    # Extract rotation (simplified - this is approximate)
    # For a more accurate conversion, we'd need to decompose the rotation matrix
    # But for now, just provide the translation
    return f"TransformationMatrix.from_xyz_rpy(x={x:.3f}, y={y:.3f}, z={z:.3f}, ...)"


def print_camera_transform(scene: trimesh.Scene):
    """Print the current camera transform in a copy-pasteable format."""
    camera_name = scene.camera.name
    if camera_name in scene.graph:
        camera_transform = scene.graph[camera_name][0]

        print("\n" + "=" * 80)
        print("CURRENT CAMERA TRANSFORM")
        print("=" * 80)
        print("\n# As numpy array:")
        print(format_transform_matrix(camera_transform))
        print(
            "\n# As TransformationMatrix (approximate - you may need to adjust yaw/pitch/roll):"
        )
        print(format_transform_xyz_rpy(camera_transform))
        print("\n" + "=" * 80)
        print(
            "Press 'P' again to print updated transform, or close the viewer to exit.\n"
        )


def main(args):
    world = None

    # Load world from obj directory if provided
    if args.obj_dir:
        print(f"Loading world from {args.obj_dir}...")
        world_loader = WarsawWorldLoader(args.obj_dir)
        world = world_loader.world
        print(f"World loaded: {world.name}")

    # Or load from database
    elif args.id is not None or args.name is not None:
        try:
            connection_string = get_connection_string()
        except EnvironmentError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        engine = create_engine(connection_string)
        session = Session(engine)

        try:
            if args.id is not None:
                print(f"Loading world with ID: {args.id}...")
                world = load_world_by_id(session, args.id)
                if world is None:
                    print(f"Error: No world found with ID '{args.id}'", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"Loading world: {args.name}...")
                world = load_world_by_name(session, args.name)
                if world is None:
                    print(
                        f"Error: No world found with name '{args.name}'",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            print(f"World loaded: {world.name}")
        finally:
            session.close()

    else:
        # List available worlds if loading from database
        try:
            connection_string = get_connection_string()
        except EnvironmentError:
            print(
                "Error: No world specified. Provide --obj-dir, --id, or --name.",
                file=sys.stderr,
            )
            print(
                "For database access, set PGDATABASE, PGUSER, and PGPASSWORD.",
                file=sys.stderr,
            )
            sys.exit(1)

        engine = create_engine(connection_string)
        session = Session(engine)

        try:
            print("Available worlds in database:")
            print("-" * 40)
            worlds = list_available_worlds(session)
            if not worlds:
                print("No worlds found in database.")
            else:
                for name, db_id in worlds:
                    print(f"  [{db_id}] {name}")
            print("-" * 40)
            print("\nUsage:")
            print("  python inspect_camera_pose.py --id <db_id>")
            print("  python inspect_camera_pose.py --name <world_name>")
            print("  python inspect_camera_pose.py <obj_dir>")
        finally:
            session.close()
        return

    # Create RayTracer scene with textures
    print("Creating scene with textures...")
    rt = RayTracer(world)
    rt.update_scene()
    scene = rt.scene

    # Set initial camera pose if provided
    if args.obj_dir:
        world_loader = WarsawWorldLoader(args.obj_dir)
        camera_poses = world_loader._predefined_camera_transforms
        if camera_poses:
            scene.camera.fov = world_loader._camera_field_of_view
            scene.graph[scene.camera.name] = camera_poses[
                0
            ]  # Use first camera pose as default

    print("\n" + "=" * 80)
    print("INTERACTIVE CAMERA VIEWER")
    print("=" * 80)
    print("Instructions:")
    print("  - Rotate: Left mouse button + drag")
    print("  - Pan: Right mouse button + drag (or Shift + Left mouse button)")
    print("  - Zoom: Mouse wheel (or Ctrl + Left mouse button)")
    print("  - Press 'P' in the viewer window to print current camera transform")
    print("  - Close window to exit")
    print("=" * 80 + "\n")

    # Create viewer and hook keyboard handler to its window
    try:
        import pyglet

        # Show scene and get the viewer object
        # scene.show() returns the viewer, but it's blocking, so we need a different approach
        # Instead, let's create the viewer manually and access its window

        from trimesh.viewer import SceneViewer

        # Try to create viewer without showing it first
        # Actually, let's use a simpler approach: monkey-patch the window creation
        original_window_init = None

        def create_viewer_with_keyboard():
            # Create a window
            window = pyglet.window.Window(
                width=1024,
                height=768,
                caption="Camera Inspector - Press 'P' to print transform",
            )

            # Store original handler if it exists
            original_key_handler = getattr(window, "on_key_press", None)

            # Create wrapper that calls both handlers
            def key_handler(symbol, modifiers):
                # Handle 'P' key first
                if symbol == pyglet.window.key.P:
                    print_camera_transform(scene)
                elif symbol == pyglet.window.key.ESCAPE:
                    window.close()

                # Call original handler if it exists
                if original_key_handler:
                    try:
                        return original_key_handler(symbol, modifiers)
                    except:
                        pass

            # Set our handler
            window.on_key_press = key_handler

            @window.event
            def on_close():
                pyglet.app.exit()

            # Now create viewer with this window
            # SceneViewer might override the handler, so we'll wrap it again after
            viewer = SceneViewer(scene, window=window)

            # After viewer is created, wrap the handler again in case it was overridden
            if hasattr(viewer, "window") and viewer.window:
                actual_window = viewer.window
            else:
                actual_window = window

            # Wrap the handler again after viewer initialization
            viewer_original_handler = getattr(actual_window, "on_key_press", None)

            def final_key_handler(symbol, modifiers):
                if symbol == pyglet.window.key.P:
                    print_camera_transform(scene)
                elif symbol == pyglet.window.key.ESCAPE:
                    actual_window.close()

                if viewer_original_handler:
                    try:
                        return viewer_original_handler(symbol, modifiers)
                    except:
                        pass

            actual_window.on_key_press = final_key_handler

            # Run event loop
            pyglet.app.run()

        create_viewer_with_keyboard()

    except Exception as e:
        import traceback

        print(f"Error setting up custom viewer: {e}")
        print(traceback.format_exc())
        # Fallback: use standard viewer and print final transform on close
        print("\nFalling back to standard viewer.")
        print(
            "After positioning camera, close viewer and final transform will be printed.\n"
        )
        scene.show()
        print("\nViewer closed. Final camera transform:")
        print_camera_transform(scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a world and interactively inspect camera poses. Press 'P' to print current camera transform."
    )
    parser.add_argument(
        "obj_dir",
        nargs="?",
        type=Path,
        help="Path to directory containing .obj files. If not provided, loads from database.",
    )
    parser.add_argument(
        "--id",
        type=int,
        help="Load world by database ID instead of obj directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Load world by name from database.",
    )

    main(parser.parse_args())
