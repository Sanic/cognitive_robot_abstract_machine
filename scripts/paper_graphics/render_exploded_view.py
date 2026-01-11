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
import trimesh
from sqlalchemy import select
from sqlalchemy.orm import Session
from krrood.ormatic.utils import create_engine

from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO, Base
from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import TriangleMesh, FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

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

    # If explosion factor is 0 or very small, return world unchanged
    if explosion_factor == 0.0 or abs(explosion_distance) < 1e-6:
        print("Explosion factor is 0, returning world unchanged")
        return exploded_world

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


def reload_textures_from_objects_dir(world: World, objects_dir: Path) -> World:
    """
    Reload meshes with textures from OBJ files in the objects directory.

    This function matches body names to OBJ files and reloads the meshes with
    textures, applying the same coordinate transformation (roll=np.pi/2) that
    WarsawWorldLoader applies during initial loading.

    Args:
        world: The world whose meshes should be reloaded
        objects_dir: Path to directory containing OBJ files

    Returns:
        The world with reloaded textures (modified in place)
    """
    if not objects_dir.exists():
        print(f"Warning: Objects directory does not exist: {objects_dir}")
        return world

    print(f"Reloading textures from {objects_dir}...")
    reloaded_count = 0

    with world.modify_world():
        for body in world.bodies:
            if not body.collision or len(body.collision) == 0:
                continue

            # Get body name (remove prefix if present)
            body_name = body.name.name
            original_body_name = body_name

            # Handle body names that might have prefixes or already include .obj extension
            if body_name.endswith(".obj"):
                # Body name already includes .obj extension, use it directly
                obj_file = objects_dir / body_name
            elif "." in body_name:
                # If name has a prefix like "prefix.name", use just "name"
                body_name = body_name.split(".")[-1]
                obj_file = objects_dir / f"{body_name}.obj"
            else:
                # No prefix, use name as-is with .obj extension
                obj_file = objects_dir / f"{body_name}.obj"

            print(
                f"  Checking body '{original_body_name}' -> looking for '{obj_file.name}' -> exists: {obj_file.exists()}"
            )

            if obj_file.exists():
                try:
                    print(f"    Loading {obj_file.name}...")
                    # Reload mesh with textures
                    # Try loading as Scene first to preserve materials, then extract mesh
                    loaded = trimesh.load(str(obj_file), process=True)

                    # If trimesh.load returns a Scene (multiple meshes), get the first mesh
                    if isinstance(loaded, trimesh.Scene):
                        geometry_names = list(loaded.geometry.keys())
                        if geometry_names:
                            reloaded_mesh = loaded.geometry[geometry_names[0]]
                            print(
                                f"    Loaded as Scene with {len(geometry_names)} geometries, using first"
                            )
                        else:
                            print(f"  Warning: No geometry found in {obj_file.name}")
                            continue
                    else:
                        reloaded_mesh = loaded
                        print(f"    Loaded as Trimesh directly")

                    # Ensure we have a Trimesh object
                    if not isinstance(reloaded_mesh, trimesh.Trimesh):
                        print(
                            f"  Warning: {obj_file.name} did not load as Trimesh (got {type(reloaded_mesh)})"
                        )
                        continue

                    # Apply the same coordinate transformation as WarsawWorldLoader:
                    # roll=np.pi/2 (90 degrees around X axis)
                    # Create a copy before transforming to preserve original
                    reloaded_mesh = reloaded_mesh.copy()
                    rotation_matrix = trimesh.transformations.rotation_matrix(
                        angle=np.pi / 2, direction=[1, 0, 0]
                    )
                    reloaded_mesh.apply_transform(rotation_matrix)

                    # Ensure visual is properly initialized
                    if (
                        not hasattr(reloaded_mesh, "visual")
                        or reloaded_mesh.visual is None
                    ):
                        reloaded_mesh.visual = trimesh.visual.TextureVisuals()

                    # Check if mesh has textures/material
                    # Check multiple ways textures might be stored
                    has_texture = False
                    texture_info = []

                    if (
                        hasattr(reloaded_mesh.visual, "material")
                        and reloaded_mesh.visual.material is not None
                    ):
                        has_texture = True
                        texture_info.append("material exists")
                        if (
                            hasattr(reloaded_mesh.visual.material, "image")
                            and reloaded_mesh.visual.material.image is not None
                        ):
                            texture_info.append("material has image")

                    # Also check if visual has UV coordinates (indicates texture mapping)
                    if (
                        hasattr(reloaded_mesh.visual, "uv")
                        and reloaded_mesh.visual.uv is not None
                    ):
                        if len(reloaded_mesh.visual.uv) > 0:
                            has_texture = True
                            texture_info.append("has UV coordinates")

                    # Debug: check what visual properties the mesh has
                    print(
                        f"    Checking {obj_file.name}: has_texture={has_texture}, visual type={type(reloaded_mesh.visual)}, info={texture_info}"
                    )
                    if hasattr(reloaded_mesh.visual, "material"):
                        print(f"      Material: {reloaded_mesh.visual.material}")
                        if hasattr(reloaded_mesh.visual.material, "image"):
                            print(
                                f"      Material has image: {reloaded_mesh.visual.material.image is not None}"
                            )

                    # Always try to reload the mesh, even if texture detection didn't work
                    # Sometimes trimesh loads textures but they're not immediately visible
                    if has_texture or True:  # Try reloading anyway
                        # Update both collision and visual shapes
                        new_collision_shapes = []
                        new_visual_shapes = []
                        updated_collision = False
                        updated_visual = False

                        # Update collision shapes
                        for collision in body.collision:
                            if (
                                isinstance(collision, FileMesh)
                                and not updated_collision
                            ):
                                # Replace FileMesh with TriangleMesh containing the reloaded mesh
                                # Make sure to preserve the visual material
                                mesh_for_collision = reloaded_mesh.copy()
                                new_collision = TriangleMesh(
                                    origin=collision.origin,
                                    scale=collision.scale,
                                    color=collision.color,
                                    mesh=mesh_for_collision,
                                )
                                # Verify material is preserved
                                if (
                                    hasattr(new_collision.mesh.visual, "material")
                                    and new_collision.mesh.visual.material is not None
                                ):
                                    print(f"    Material preserved in collision mesh")
                                new_collision_shapes.append(new_collision)
                                updated_collision = True
                            elif (
                                isinstance(collision, TriangleMesh)
                                and not updated_collision
                            ):
                                # Update TriangleMesh directly
                                collision.mesh = reloaded_mesh.copy()
                                new_collision_shapes.append(collision)
                                updated_collision = True
                            else:
                                # Keep other collision shapes as-is
                                new_collision_shapes.append(collision)

                        # Update visual shapes
                        for visual in body.visual:
                            if isinstance(visual, FileMesh) and not updated_visual:
                                # Replace FileMesh with TriangleMesh containing the reloaded mesh
                                mesh_for_visual = reloaded_mesh.copy()
                                new_visual = TriangleMesh(
                                    origin=visual.origin,
                                    scale=visual.scale,
                                    color=visual.color,
                                    mesh=mesh_for_visual,
                                )
                                # Verify material is preserved
                                if (
                                    hasattr(new_visual.mesh.visual, "material")
                                    and new_visual.mesh.visual.material is not None
                                ):
                                    print(f"    Material preserved in visual mesh")
                                new_visual_shapes.append(new_visual)
                                updated_visual = True
                            elif (
                                isinstance(visual, TriangleMesh) and not updated_visual
                            ):
                                # Update TriangleMesh directly
                                visual.mesh = reloaded_mesh.copy()
                                new_visual_shapes.append(visual)
                                updated_visual = True
                            else:
                                # Keep other visual shapes as-is
                                new_visual_shapes.append(visual)

                        if updated_collision:
                            # Create new ShapeCollection with updated shapes
                            body.collision = ShapeCollection(new_collision_shapes)
                        if updated_visual:
                            # Create new ShapeCollection with updated visual shapes
                            body.visual = ShapeCollection(new_visual_shapes)

                        if updated_collision or updated_visual:
                            reloaded_count += 1
                            texture_status = (
                                "with textures"
                                if has_texture
                                else "without detected textures"
                            )
                            print(
                                f"  Reloaded {body_name}.obj from {obj_file.name} {texture_status}"
                            )
                    else:
                        print(
                            f"  Warning: {obj_file.name} has no textures (visual type: {type(reloaded_mesh.visual)})"
                        )

                except Exception as e:
                    print(f"  Error reloading {obj_file.name}: {e}")
                    continue
            else:
                # Try alternative names (e.g., with different case or extensions)
                # Look for any OBJ file that might match
                matching_files = list(objects_dir.glob(f"{body_name}*.obj"))
                if matching_files:
                    obj_file = matching_files[0]
                    try:
                        reloaded_mesh = trimesh.load(
                            str(obj_file), process=True, force="mesh"
                        )
                        if isinstance(reloaded_mesh, trimesh.Scene):
                            geometry_names = list(reloaded_mesh.geometry.keys())
                            if geometry_names:
                                reloaded_mesh = reloaded_mesh.geometry[
                                    geometry_names[0]
                                ]
                            else:
                                continue

                        rotation_matrix = trimesh.transformations.rotation_matrix(
                            angle=np.pi / 2, direction=[1, 0, 0]
                        )
                        reloaded_mesh.apply_transform(rotation_matrix)

                        has_texture = (
                            hasattr(reloaded_mesh.visual, "material")
                            and reloaded_mesh.visual.material is not None
                        )

                        if has_texture:
                            # Update the mesh in the collision shape
                            new_collision_shapes = []
                            updated = False
                            for collision in body.collision:
                                if isinstance(collision, FileMesh) and not updated:
                                    # Replace FileMesh with TriangleMesh containing the reloaded mesh
                                    new_collision = TriangleMesh(
                                        origin=collision.origin,
                                        scale=collision.scale,
                                        color=collision.color,
                                        mesh=reloaded_mesh,
                                    )
                                    new_collision_shapes.append(new_collision)
                                    updated = True
                                    reloaded_count += 1
                                    print(
                                        f"  Reloaded {body_name}.obj from {obj_file.name} with textures"
                                    )
                                elif (
                                    isinstance(collision, TriangleMesh) and not updated
                                ):
                                    # Update TriangleMesh directly
                                    collision.mesh = reloaded_mesh
                                    new_collision_shapes.append(collision)
                                    updated = True
                                    reloaded_count += 1
                                    print(
                                        f"  Reloaded {body_name}.obj from {obj_file.name} with textures"
                                    )
                                else:
                                    # Keep other collision shapes as-is
                                    new_collision_shapes.append(collision)

                            if updated:
                                # Create new ShapeCollection with updated shapes
                                body.collision = ShapeCollection(new_collision_shapes)
                    except Exception as e:
                        print(f"  Error reloading {obj_file.name}: {e}")
                        continue

    print(f"Reloaded textures for {reloaded_count} bodies.")
    return world


def render_exploded_world(
    world: World,
    save_path: Optional[Path] = None,
    show: bool = True,
    camera_index: int = 0,
    explosion_factor: float = 1.0,
    objects_dir: Optional[Path] = None,
) -> bytes:
    """
    Render the world in an exploded view.

    Args:
        world: The world to render
        save_path: Optional path to save the rendered image
        show: Whether to show the interactive viewer
        camera_index: Index of predefined camera pose (0-3)
        explosion_factor: Multiplier for explosion distance
        objects_dir: Optional path to directory containing OBJ files for texture reloading

    Returns:
        PNG image data as bytes
    """
    # Create exploded version of the world
    print("Creating exploded view...")
    exploded_world = create_exploded_world(world, explosion_factor=explosion_factor)

    # Reload textures from Objects directory if provided
    # Do this AFTER creating exploded world to ensure textures are in the final world
    if objects_dir is not None:
        exploded_world = reload_textures_from_objects_dir(exploded_world, objects_dir)

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
                objects_dir=args.objects_dir,
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
    parser.add_argument(
        "--objects-dir",
        type=Path,
        help="Path to directory containing OBJ files for texture reloading. If provided, meshes will be reloaded with textures from this directory.",
    )

    main(parser.parse_args())
