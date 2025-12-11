import argparse
from pathlib import Path
from typing import List
import requests
import json
import os
import base64

# Import shared functions from load_warsaw_scene
import sys

from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader

sys.path.insert(
    0, str(Path(__file__).parent.parent / "semantic_digital_twin" / "scripts")
)

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def query_vlm(
    original_image_bytes: bytes,
    highlighted_image_bytes: bytes,
    object_taxonomy: str,
    spatial_relations: str,
    color_names: List[str],
    semantic_labels: str,
) -> dict:
    """Query the VLM with original and highlighted images."""
    base64_original = encode_image_bytes(original_image_bytes)
    base64_highlighted = encode_image_bytes(highlighted_image_bytes)

    system_prompt = f"""You are a semantic perception system for robotic scene understanding.

## Your Task
Analyze images and classify objects according to a given ontology. You will receive:
1. An image of a scene with original textures
2. An image with specific objects highlighted in distinct colors
3. The prior from a previous semantic segmentation step for each highlighted object

Focus ONLY on the highlighted objects. For each:
- Identify its class from the provided taxonomy
- If no suitable class exists, propose a new subclass under the most appropriate parent

### Important rules to keep in mind
1. A class cannot be its own superclass
2. The prior semantic segmentation may be wrong. In such cases, please provide a correct semantic class name.

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

Image 1: Original scene with natural textures
Image 2: Same scene with target objects highlighted in distinct colors

Identify the highlighted objects.
"""

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
                "model": "qwen/qwen2.5-vl-72b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_original}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_highlighted}"
                                },
                            },
                        ],
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

    # Load world
    print(f"Loading world from {obj_dir}...")

    world_loader = WarsawWorldLoader(obj_dir)
    world = world_loader.world
    bodies = world.bodies_with_enabled_collision

    # Export semantic annotations JSON for VLM context
    world_loader.export_semantic_annotation_inheritance_structure(export_path)

    # Read taxonomy and spatial relations
    object_taxonomy = (export_path / "semantic_annotations.json").read_text()
    spatial_relations_path = export_path / "spatial_relations.txt"
    if spatial_relations_path.exists():
        spatial_relations = spatial_relations_path.read_text()
    else:
        spatial_relations = "on, under, next_to, inside, above, below, left_of, right_of, in_front_of, behind"

    if not args.skip_vlm:
        # Set up visual state management
        camera_pose = world_loader._predefined_camera_transforms[
            1
        ]  # Use first camera pose for VLM queries

        # Render original scene once
        print("Rendering original scene...")
        original_image = world_loader.render_scene_from_camera_pose(
            camera_pose, export_path / "scene_orig.png"
        )

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

            # Render highlighted scene
            highlighted_image = world_loader.render_scene_from_camera_pose(
                camera_pose, export_path / f"scene_{i}.png"
            )

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
                original_image,
                highlighted_image,
                object_taxonomy,
                spatial_relations,
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

    else:  # Skipped VLM: Read from fil instead
        with open(output_file, "r") as f:
            all_responses = json.load(f)

    # Extract and save summary
    summary = extract_summary(all_responses)
    summary_file = output_file.parent / (output_file.stem + "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")

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
    parser.add_argument("--skip_vlm", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
