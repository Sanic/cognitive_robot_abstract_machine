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
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session
from krrood.ormatic.utils import create_engine

# Import ORM interface - this brings in all DAOs including WorldMappingDAO
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO, Base
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


def render_world(
    world: World, save_path: Optional[Path] = None, show: bool = True
) -> bytes:
    """
    Render the world using RayTracer.

    Args:
        world: The world to render
        save_path: Optional path to save the rendered image
        show: Whether to show the interactive viewer

    Returns:
        PNG image data as bytes
    """
    rt = RayTracer(world)
    rt.update_scene()

    # Save image if path provided
    if save_path:
        png_data = rt.scene.save_image(resolution=(1920, 1080), visible=True)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(png_data)
        print(f"Image saved to: {save_path}")

    # Show interactive viewer
    if show:
        print("Opening interactive viewer...")
        rt.scene.show()

    return (
        rt.scene.save_image(resolution=(1920, 1080), visible=True) if save_path else b""
    )


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
            render_world(world, save_path=args.save)

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

    main(parser.parse_args())
