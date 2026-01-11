import argparse
from pathlib import Path
from typing import Dict, List
import requests
import json
import os
import base64
import numpy as np
import trimesh

# Import shared functions from load_warsaw_scene
import sys

from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader
from semantic_digital_twin.spatial_types import TransformationMatrix

from sqlalchemy import select
from sqlalchemy.orm import Session
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.orm.ormatic_interface import Base, WorldMappingDAO
from krrood.ormatic.dao import to_dao

sys.path.insert(
    0, str(Path(__file__).parent.parent / "semantic_digital_twin" / "scripts")
)

DB_NAME = os.getenv("PGDATABASE")
DB_USER = os.getenv("PGUSER")
DB_PASSWORD = os.getenv("PGPASSWORD")

DB_HOST = "localhost"
DB_PORT = 5432

connection_string = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def create_camera_poses() -> Dict[str, np.ndarray]:
    """
    Create camera transforms
    """
    return {
        "back": np.array(
            [
                [0.019661, 0.565278, -0.824666, -3.302801],
                [-0.999801, 0.008306, -0.018143, -0.113984],
                [-0.003406, 0.824859, 0.565329, 2.521490],
                [0.000000, 0.000000, 0.000000, 1.000000],
            ]
        ),
        "front_right": np.array(
            [
                [0.493480, -0.501933, 0.710310, 2.967743],
                [0.863728, 0.378807, -0.332385, -1.558169],
                [-0.102235, 0.777540, 0.620467, 2.430472],
                [0.000000, 0.000000, 0.000000, 1.000000],
            ]
        ),
        "front_left": np.array(
            [
                [-0.658764, -0.383959, 0.646997, 2.903789],
                [0.752266, -0.323319, 0.574074, 3.060092],
                [-0.011235, 0.864893, 0.501830, 2.281899],
                [0.000000, 0.000000, 0.000000, 1.000000],
            ]
        ),
    }


def query_vlm(
    original_images: List[bytes],
    highlighted_images: List[bytes],
    object_taxonomy: str,
    color_names: List[str],
    semantic_labels: str,
) -> dict:
    """Query the VLM with original and highlighted images from multiple viewpoints."""
    base64_originals = [encode_image_bytes(img) for img in original_images]
    base64_highlighted = [encode_image_bytes(img) for img in highlighted_images]

    system_prompt = f"""You are a semantic perception system for robotic scene understanding.

## Your Task
Analyze images and classify objects according to a given ontology. You will receive:
1. Three images of a scene with original textures from different viewpoints (diagonal front left, diagonal front right, back)
2. Three corresponding images with specific objects highlighted in distinct colors from the same viewpoints
3. The prior from a previous semantic segmentation step for each highlighted object

Focus ONLY on the highlighted objects. For each:
- Identify its class from the provided taxonomy
- If no suitable class exists, propose a new subclass under the most appropriate parent

### Important rules to keep in mind
1. A class cannot be its own superclass
2. The prior semantic segmentation may be wrong. In such cases, please provide a correct semantic class name.
3. Use all three viewpoints to get a complete understanding of each object's shape and context.

## Output Schema
Respond with valid JSON:
{{
  "objects": [
    {{
      "highlight_color": "string (the color used to highlight this object: one of {color_names})",
      "classification": {{
        "class": "string (class name from taxonomy, or your proposed new class)",
        "superclass": "string (parent class in taxonomy)",
        "is_new_class": "boolean (true if you're proposing a new class)",
        "new_class_justification": "string | null (if is_new_class, explain why existing classes don't fit)"
      }},
      "confidence": "number (0-1)"
    }}
  ],
  "notes": "string | null (any ambiguities or uncertainties)"
}}"""

    user_prompt = f"""## Object Taxonomy (Hierarchy)
{object_taxonomy}

## Prior semantic labels
{semantic_labels}

## Images

Images 1-3: Original scene with natural textures from three viewpoints:
  - Image 1: Back view
  - Image 2: Diagonal front right view
  - Image 3: Diagonal front left view

Images 4-6: Same scene with target objects highlighted in distinct colors from the same viewpoints:
  - Image 4: Back view (highlighted)
  - Image 5: Diagonal front right view (highlighted)
  - Image 6: Diagonal front left view (highlighted)

Identify the highlighted objects using all viewpoints for a complete understanding.
"""

    # Build content array with text and all images
    content = [{"type": "text", "text": user_prompt}]

    # Add original images
    for base64_img in base64_originals:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"},
            }
        )

    # Add highlighted images
    for base64_img in base64_highlighted:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"},
            }
        )

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "ai.uni-bremen.de",
            "X-Title": "Uni Bremen",
        },
        data=json.dumps(
            {
                "model": "qwen/qwen3-vl-30b-a3b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
            }
        ),
    )
    return response.json()


def extract_summary(all_responses: List[dict]) -> List[dict]:
    """
    Extract a summarized form from the raw VLM responses.
    For each object, returns: body_id, color, class, superclass, is_new_class, confidence.
    """
    summary = []

    for group_data in all_responses:
        body_ids = group_data["body_ids"]
        colors = group_data["colors"]
        vlm_response = group_data["vlm_response"]

        # Parse VLM content (may be wrapped in markdown code block)
        try:
            content = vlm_response["choices"][0]["message"]["content"]
            # Remove markdown code block if present
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            objects = parsed.get("objects", [])
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            print(
                f"Warning: Could not parse VLM response for group {group_data['group_index']}: {e}"
            )
            continue

        for i, obj in enumerate(objects):
            color_vlm = obj.get("highlight_color", "")
            body_id = body_ids[colors.index(color_vlm)]

            classification = obj.get("classification", {})

            summary.append(
                {
                    "body_id": body_ids[i],
                    "color": color_vlm,
                    "class": classification.get("class"),
                    "superclass": classification.get("superclass"),
                    "is_new_class": classification.get("is_new_class", False),
                    "confidence": obj.get("confidence"),
                }
            )

    return summary


def main(args):
    obj_dir = args.obj_dir
    output_file = args.output_file
    group_size = args.group_size
    export_path = Path(args.export_dir)

    # Check API key if VLM will be called
    if not args.render_only and not args.skip_vlm:
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. Required for VLM queries."
            )

    # Path for rendered images - save to scripts/model_input/
    script_dir = Path(__file__).parent
    model_input_dir = script_dir / "model_input"
    model_input_dir.mkdir(parents=True, exist_ok=True)

    # Create database engine and tables for later persistence of the world
    engine = create_engine(connection_string, echo=True)  # echo=True for debugging SQL
    Base.metadata.create_all(bind=engine)
    print("Database tables created")

    # Load world
    print(f"Loading world from {obj_dir}...")

    world_loader = WarsawWorldLoader(obj_dir)
    world = world_loader.world
    bodies = world.bodies_with_enabled_collision

    # Export semantic annotations JSON for VLM context
    world_loader.export_semantic_annotation_inheritance_structure(export_path)

    # Read taxonomy
    object_taxonomy = (export_path / "semantic_annotations.json").read_text()

    # Create custom diagonal camera transforms (returns dict)
    camera_poses_dict = create_camera_poses()

    if not args.skip_vlm and not args.render_only:
        # Render original scene from 3 viewpoints
        print("Rendering original scene from 3 viewpoints...")
        original_images = []
        for pose_name, camera_pose in camera_poses_dict.items():
            print(f"  Rendering {pose_name} view...")
            image_bytes = world_loader.render_scene_from_camera_pose(
                camera_pose, model_input_dir / f"scene_orig_{pose_name}.png"
            )
            original_images.append(image_bytes)

        # Process groups
        all_responses = []
        num_groups = (len(bodies) + group_size - 1) // group_size

        for i, start in enumerate(range(0, len(bodies), group_size)):
            group = bodies[start : start + group_size]
            print(f"Processing group {i + 1}/{num_groups} ({len(group)} objects)...")

            # Reset visuals and apply highlight colors
            world_loader._reset_body_colors()
            bodies_colors = world_loader._apply_highlight_to_group(group)
            color_names = list(
                map(lambda c: c.closest_css3_color_name(), bodies_colors.values())
            )

            # Render highlighted scene from 3 viewpoints
            print(f"  Rendering highlighted scene from 3 viewpoints...")
            highlighted_images = []
            for pose_name, camera_pose in camera_poses_dict.items():
                image_bytes = world_loader.render_scene_from_camera_pose(
                    camera_pose, model_input_dir / f"scene_{i}_{pose_name}.png"
                )
                highlighted_images.append(image_bytes)

            semantic_labels_dict = {
                body_id: color_names[i]
                for i, body_id in enumerate(bodies_colors.keys())
            }
            semantic_labels = ""
            for body_uuid, color_name in semantic_labels_dict.items():
                body = next(filter(lambda b: b.id == body_uuid, bodies))
                semantic_labels += f"{color_name}: {body.name}\n"

            # Query VLM
            print(f"  Querying VLM for group {i + 1}...")
            response = query_vlm(
                original_images,
                highlighted_images,
                object_taxonomy,
                color_names,
                semantic_labels,
            )
            print(f"  Response: {response}")

            all_responses.append(
                {
                    "group_index": i,
                    "body_ids": list(map(str, bodies_colors.keys())),
                    "colors": color_names,
                    "vlm_response": response,
                }
            )

            # Reset for next iteration
            world_loader._reset_body_colors()

        # Save raw responses
        with open(output_file, "w") as f:
            json.dump(all_responses, f, indent=2)
        print(f"Raw results saved to {output_file}")

        # Extract and save summary
        summary = extract_summary(all_responses)
        summary_file = output_file.parent / (output_file.stem + "_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

    elif args.render_only:
        # Render-only mode: just save images without calling VLM
        print("Render-only mode: Saving images without VLM queries...")

        # Render original scene from 3 viewpoints
        print("Rendering original scene from 3 viewpoints...")
        for idx, camera_pose in enumerate(camera_poses):
            print(f"  Rendering {camera_names[idx]} view...")
            world_loader.render_scene_from_camera_pose(
                camera_pose, model_input_dir / f"scene_orig_{camera_names[idx]}.png"
            )

        # Process groups
        num_groups = (len(bodies) + group_size - 1) // group_size

        for i, start in enumerate(range(0, len(bodies), group_size)):
            group = bodies[start : start + group_size]
            print(f"Processing group {i + 1}/{num_groups} ({len(group)} objects)...")

            # Reset visuals and apply highlight colors
            world_loader._reset_body_colors()
            bodies_colors = world_loader._apply_highlight_to_group(group)
            color_names = list(
                map(lambda c: c.closest_css3_color_name(), bodies_colors.values())
            )

            # Render highlighted scene from 3 viewpoints
            print(f"  Rendering highlighted scene from 3 viewpoints...")
            for idx, camera_pose in enumerate(camera_poses):
                world_loader.render_scene_from_camera_pose(
                    camera_pose, model_input_dir / f"scene_{i}_{camera_names[idx]}.png"
                )

            # Reset for next iteration
            world_loader._reset_body_colors()

        print(f"All images saved to {model_input_dir}")
        all_responses = []
        summary = []

    elif args.skip_vlm:  # Skipped VLM: Read from file instead
        with open(output_file, "r") as f:
            all_responses = json.load(f)

        # Extract and save summary
        summary = extract_summary(all_responses)
        summary_file = output_file.parent / (output_file.stem + "_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

    # Convert world to DAO and persist
    with Session(engine) as session:
        world_dao: WorldMappingDAO = to_dao(world)
        session.add(world_dao)
        session.commit()
        print(f"World persisted with database_id: {world_dao.database_id}")

    return all_responses, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query VLM for scene understanding")
    parser.add_argument(
        "obj_dir", type=Path, help="Path to directory containing .obj files"
    )
    parser.add_argument("output_file", type=Path, help="Path to output JSON file")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("./vlm_export"),
        help="Directory for exported metadata",
    )
    parser.add_argument(
        "--group-size", type=int, default=8, help="Number of objects per group"
    )
    parser.add_argument(
        "--skip_vlm",
        action="store_true",
        default=False,
        help="Skip VLM queries and read responses from file instead",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        default=False,
        help="Only render images without calling VLM. Images are saved to scripts/model_input/",
    )
    args = parser.parse_args()
    main(args)
