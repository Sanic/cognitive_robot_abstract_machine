"""
Script to refine class structure from VLM classification summary.

Takes a summary JSON (body_id -> class mapping) and:
1. Creates new classes if they don't exist
2. Inspects class constructors to find required slots
3. Queries VLM to determine slot fillers from other detected objects
"""

import argparse
import json
import logging
import os
import base64
from pathlib import Path
from typing import Dict, List, Any, Tuple, Type, Optional
import requests

from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO
from semantic_digital_twin.semantic_annotations import semantic_annotations as sa_module
from semantic_digital_twin.semantic_annotations.mixins import HasBody
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationClassBuilder,
    SemanticAnnotationFilePaths,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)
from semantic_digital_twin.utils import InheritanceStructureExporter
from semantic_digital_twin.orm.ormatic_interface import Base

from sqlalchemy import select
from sqlalchemy.orm import Session
from krrood.ormatic.utils import create_engine

DB_NAME = os.getenv("PGDATABASE")
DB_USER = os.getenv("PGUSER")
DB_PASSWORD = os.getenv("PGPASSWORD")

DB_HOST = "localhost"
DB_PORT = 5432

# Create PostgreSQL connection string
connection_string = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Create exporter instance to use its collect_required_public_fields method
_field_exporter = InheritanceStructureExporter(root_class=SemanticAnnotation)


def build_class_lookup() -> Dict[str, Type]:
    """Build a name -> class lookup from the semantic_annotations module."""
    lookup = {}
    for name in dir(sa_module):
        obj = getattr(sa_module, name)
        if isinstance(obj, type):
            lookup[name] = obj
    return lookup


def get_class_constructor_fields(cls: Type) -> List[Dict[str, Any]]:
    """
    Get required public fields from a class's constructor.
    Returns a list of dicts with field 'name' and 'type'.
    """
    return _field_exporter.collect_required_public_fields(cls)


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def query_cheap_llm_for_candidates(
    target_object: Dict[str, Any],
    target_class: Type,
    complex_fields: List[Dict[str, Any]],
    all_objects: List[Dict[str, Any]],
    class_lookup: Dict[str, Type],
    max_candidates: int = 8,
) -> Dict[str, List[str]]:
    """
    Query a cheap LLM to narrow down candidate objects for each slot.
    Returns a dict mapping field_name -> list of candidate body_ids.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set, cannot query LLM")

    prompt = f"""You are helping to instantiate semantic annotations for a robotic scene understanding system.

## Target Object
- Body ID: {target_object['body_id']}
- Detected Class: {target_object['class']}

## Constructor Slots to Fill
The class `{target_class.__name__}` requires the following fields:
"""
    all_compatible_classes = set()
    for field in complex_fields:
        field_type = field["type"]
        # Get compatible classes (the field type and its subclasses)
        compatible_classes = [field_type]
        if field_type in class_lookup:
            base_cls = class_lookup[field_type]
            for name, cls in class_lookup.items():
                if (
                    isinstance(cls, type)
                    and issubclass(cls, base_cls)
                    and name != field_type
                ):
                    compatible_classes.append(name)
        prompt += f"- `{field['name']}`: expects `{field_type}` (compatible classes: {', '.join(compatible_classes)})\n"
        all_compatible_classes.update(compatible_classes)

    prompt += f"""
## Available Objects in Scene
"""
    candidate_obj_info = []
    for obj in all_objects:
        if (
            obj["body_id"] != target_object["body_id"]
            and obj["class"] in all_compatible_classes
        ):  # Exclude target itself
            candidate_obj_info.append(
                {
                    "body_id": obj["body_id"],
                    "class": obj.get("class", "Unknown"),
                    "confidence": obj.get("confidence", 0),
                }
            )
        prompt += f"- body_id: {obj['body_id']}, class: {obj['class']}, confidence: {obj['confidence']}\n"

    prompt += f"""
## Your Task
For each slot, select up to {max_candidates} candidate objects that could potentially fill it.
Consider:
1. Type compatibility (the object's class must match or be a subclass of the required type)
2. Semantic plausibility (e.g., a Drawer's container is likely to be a Cabinet, a Door's handle is likely a Handle)

Respond with valid JSON:
{{
  "candidates": {{
    "<field_name>": ["body_id_1", "body_id_2", ...],
    ...
  }},
  "reasoning": "brief explanation of selection criteria"
}}
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
                "model": "mistralai/devstral-2512:free",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a semantic scene understanding assistant. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
        ),
    )

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        candidates = parsed.get("candidates", {})
        logging.info(f"Cheap LLM selected candidates: {candidates}")
        logging.info(f"Reasoning: {parsed.get('reasoning', 'N/A')}")
        return candidates
    except Exception as e:
        logging.error(f"Failed to parse cheap LLM response: {e}")
        # Fallback: return all objects as candidates for each field
        all_body_ids = [obj["body_id"] for obj in candidate_obj_info]
        return {
            field["name"]: all_body_ids[:max_candidates] for field in complex_fields
        }


def generate_slot_filling_prompt(
    target_object: Dict[str, Any],
    target_class: Type,
    complex_fields: List[Dict[str, Any]],
    candidate_objects: List[Dict[str, Any]],
    color_mapping: Dict[str, str],
) -> Tuple[str, str]:
    """
    Generate a prompt for the VLM to determine slot fillers for a semantic annotation.

    Args:
        target_object: The object that needs its slots filled
        target_class: The class type of the target object
        complex_fields: List of fields that need to be filled
        candidate_objects: Pre-filtered list of candidate objects (from cheap LLM)
        color_mapping: Dict mapping body_id -> highlight color in the rendered image
    """
    if not complex_fields:
        return None, None

    system_prompt = f"""You are helping to instantiate semantic annotations for a robotic scene understanding system.

## Target Object
- Body ID: {target_object['body_id']}
- Detected Class: {target_object['class']}
- Highlight Color: {color_mapping[str(target_object['body_id'])]}

## Constructor Signature
The class `{target_class.__name__}` requires the following fields to be filled:
"""

    for field in complex_fields:
        system_prompt += (
            f"- `{field['name']}`: expects an instance of `{field['type']}`\n"
        )

    system_prompt += f"""
## Candidate Objects
The following objects are highlighted with distinct colors in the provided image:
"""
    for obj in candidate_objects:
        body_id = obj["body_id"]
        color = color_mapping.get(body_id, "unknown")
        system_prompt += f"- body_id: {body_id}, class: {obj.get('class', 'Unknown')}, highlight_color: {color}\n"

    logging.info(system_prompt)

    user_prompt = f"""
## Your Task
Look at the candidate objects in the image. For each required field in the constructor of the target object, determine which candidate object should fill that slot.
Consider spatial relationships visible in the image (e.g., a Drawer typically belongs to a Cabinet or Dresser, a Handle belongs to a Drawer or Door).

Respond with valid JSON:
{{
  "slot_assignments": [
    {{
      "field_name": "string (the field name from the constructor)",
      "field_type": "string (the expected type)",
      "assigned_body_id": "string (body_id of the object to use, or null if cannot be determined)",
      "highlight_color": "string (the color of the assigned object in the image)",
      "reasoning": "string (brief explanation of why this assignment makes sense based on the image)"
    }}
  ],
  "notes": "string | null (any issues or ambiguities)"
}}
"""
    logging.info(user_prompt)
    return system_prompt, user_prompt


def query_vlm_for_slots(
    system_prompt: str, user_prompt: str, scene_image: bytes
) -> Dict[str, Any]:
    """Query the VLM to determine slot fillers."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set, cannot query VLM")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    base64_image = encode_image_bytes(scene_image)
    messages[1]["content"].append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
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
                "model": "qwen/qwen2.5-vl-72b-instruct",
                "messages": messages,
            }
        ),
    )

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        logging.error(f"Failed to parse VLM response: {e}")
        return {"slot_assignments": [], "notes": f"Parse error: {e}"}


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load world
    logging.info(f"Loading world from database...")
    engine = create_engine(connection_string, echo=True)  # echo=True for debugging SQL
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        queried_dao = session.scalar(
            select(WorldMappingDAO).where(
                WorldMappingDAO.database_id == args.world_database_id
            )
        )
        if queried_dao:
            world: World = queried_dao.from_dao()
    world_loader = WarsawWorldLoader.from_world(world)

    # Load summary
    with open(args.summary_json, "r") as f:
        summary: List[Dict[str, Any]] = json.load(f)

    logging.info(f"Loaded {len(summary)} object classifications")

    # Build class lookup
    class_lookup = build_class_lookup()

    # Phase 1: Create any missing classes
    for obj in summary:
        if obj.get("confidence", 0) < args.min_confidence:
            logging.warning(
                f"Skipping {obj['body_id']} ({obj['class']}) due to low confidence"
            )
            continue

        cls_name = obj["class"]
        superclass_name = obj.get("superclass", "SemanticAnnotation")

        if cls_name not in class_lookup:
            logging.info(
                f"Creating new class: {cls_name} (superclass: {superclass_name})"
            )

            if superclass_name not in class_lookup:
                logging.warning(
                    f"Superclass '{superclass_name}' not found, using SemanticAnnotation"
                )
                superclass_name = "SemanticAnnotation"

            superclass = class_lookup[superclass_name]

            builder = SemanticAnnotationClassBuilder(
                cls_name, template_name="dataclass_template.py.jinja"
            )
            cls = builder.add_base(superclass).build()
            builder.append_to_file(
                SemanticAnnotationFilePaths.MAIN_SEMANTIC_ANNOTATION_FILE.value,
                include_imports=True,
            )
            class_lookup[cls_name] = cls
            logging.info(f"Created class '{cls_name}'")

    # Phase 2: Analyze constructor requirements and generate slot-filling prompts
    slot_filling_results = []

    # Build body_id -> object lookup for easy access
    body_id_to_obj = {obj["body_id"]: obj for obj in summary}

    for obj in summary:
        if obj.get("confidence", 0) < args.min_confidence:
            continue

        cls_name = obj["class"]
        if cls_name not in class_lookup:
            raise RuntimeError(
                f"Class '{cls_name}' not found. Class should have been created in previous step"
            )

        cls = class_lookup[cls_name]
        fields = get_class_constructor_fields(cls)

        # Check if this class needs slot filling (all fields from collect_required_public_fields are required)
        # Only consider fields whose type is a SemanticAnnotation subclass
        complex_fields = [
            f
            for f in fields
            if f["name"] != "body"
            and f["type"] in class_lookup
            and issubclass(class_lookup[f["type"]], SemanticAnnotation)
        ]

        if complex_fields:
            logging.info(
                f"Object {obj['body_id']} ({cls_name}) needs slot filling for: {[f['name'] for f in complex_fields]}"
            )

            # Step 2a: Query cheap LLM to narrow down candidates
            logging.info(f"Querying cheap LLM for candidate selection...")
            candidates_by_field = query_cheap_llm_for_candidates(
                target_object=obj,
                target_class=cls,
                complex_fields=complex_fields,
                all_objects=summary,
                class_lookup=class_lookup,
                max_candidates=args.max_candidates,
            )

            # Collect all unique candidate body_ids across all fields
            all_candidate_ids = set()
            for field_candidates in candidates_by_field.values():
                all_candidate_ids.update(field_candidates)

            # Get candidate objects
            candidate_objects = [
                body_id_to_obj[bid]
                for bid in all_candidate_ids
                if bid in body_id_to_obj
            ]

            if not candidate_objects:
                logging.warning(
                    f"No valid candidates found for {obj['body_id']}, skipping VLM query"
                )
                continue

            logging.info(
                f"Filtered to {len(candidate_objects)} candidate objects for VLM query"
            )

            # Step 2b: Render scene with only candidate objects and target object highlighted
            # Get bodies for candidates and apply highlight colors
            highlighted_bodies = []
            for cand_obj in candidate_objects:
                try:
                    body = next(
                        b for b in world.bodies if str(b.id) == cand_obj["body_id"]
                    )
                    highlighted_bodies.append(body)
                except StopIteration:
                    raise RuntimeError(
                        f"Body not found for candidate {cand_obj['body_id']}"
                    )

            # Add target obj to highlighted bodies
            highlighted_bodies.append(
                next(b for b in world.bodies if str(b.id) == obj["body_id"])
            )

            if highlighted_bodies:
                # Reset colors and apply highlights to candidates
                world_loader._reset_body_colors()
                bodies_colors = world_loader._apply_highlight_to_group(
                    highlighted_bodies
                )

                # Build color mapping: body_id -> color name
                color_mapping = {
                    str(body_id): color.closest_css3_color_name()
                    for body_id, color in bodies_colors.items()
                }

                # Render scene with highlighted candidates
                camera_pose = world_loader._predefined_camera_transforms[1]
                scene_image = world_loader.render_scene_from_camera_pose(camera_pose)

                # Reset colors after rendering
                world_loader._reset_body_colors()
            else:
                raise RuntimeError(
                    "Would be using VLM without scene image. Something likely went wrong"
                )

            # Step 2c: Generate prompt and query VLM
            system_prompt, user_prompt = generate_slot_filling_prompt(
                target_object=obj,
                target_class=cls,
                complex_fields=complex_fields,
                candidate_objects=candidate_objects,
                color_mapping=color_mapping,
            )

            if system_prompt and user_prompt and not args.dry_run and scene_image:
                logging.info(f"Querying VLM for slot assignments...")
                vlm_result = query_vlm_for_slots(
                    system_prompt, user_prompt, scene_image
                )

                slot_filling_results.append(
                    {
                        "body_id": obj["body_id"],
                        "class": cls_name,
                        "required_fields": complex_fields,
                        "candidates_by_field": candidates_by_field,
                        "color_mapping": color_mapping,
                        "vlm_response": vlm_result,
                    }
                )
            elif system_prompt and user_prompt and args.dry_run:
                # Dry run: just save the prompt
                slot_filling_results.append(
                    {
                        "body_id": obj["body_id"],
                        "class": cls_name,
                        "required_fields": complex_fields,
                        "candidates_by_field": candidates_by_field,
                        "color_mapping": color_mapping,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    }
                )
            else:
                raise RuntimeError(
                    f"No prompt was created for object {obj['body_id']} of class {obj['class']}. This points to a deeper problem."
                )

    # Save results
    output_file = (
        args.output_file or args.summary_json.parent / "slot_filling_results.json"
    )
    with open(output_file, "w") as f:
        json.dump(slot_filling_results, f, indent=2)
    logging.info(f"Slot filling results saved to {output_file}")

    # Phase 3: Instantiate objects (if not dry run)
    if not args.dry_run:
        instantiated = []
        body_id_to_instance = {}

        # First pass: instantiate simple objects (HasBody subclasses with just a body field)
        for obj in summary:
            if obj.get("confidence", 0) < args.min_confidence:
                continue

            cls_name = obj["class"]
            if cls_name not in class_lookup:
                continue

            cls = class_lookup[cls_name]
            fields = get_class_constructor_fields(cls)

            # Check if it's a simple HasBody subclass (all fields from collect_required_public_fields are required)
            if len(fields) == 1 and fields[0]["name"] == "body":
                try:
                    body = next(b for b in world.bodies if str(b.id) == obj["body_id"])
                    instance = cls(body=body)
                    body_id_to_instance[obj["body_id"]] = instance
                    instantiated.append(
                        {
                            "body_id": obj["body_id"],
                            "class": cls_name,
                            "status": "created",
                        }
                    )
                    logging.info(
                        f"Created {cls_name} instance for body {obj['body_id']}"
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to create {cls_name} for {obj['body_id']}: {e}"
                    )

        # Second pass: instantiate complex objects using slot filling results
        for result in slot_filling_results:
            if "vlm_response" not in result:
                continue

            cls_name = result["class"]
            cls = class_lookup[cls_name]
            body_id = result["body_id"]

            try:
                kwargs = {}

                # Get body if needed
                if issubclass(cls, HasBody):
                    body = next(b for b in world.bodies if str(b.id) == body_id)
                    kwargs["body"] = body

                # Fill slots from VLM response
                for assignment in result["vlm_response"].get("slot_assignments", []):
                    field_name = assignment["field_name"]
                    assigned_id = assignment.get("assigned_body_id")

                    if assigned_id and assigned_id in body_id_to_instance:
                        kwargs[field_name] = body_id_to_instance[assigned_id]
                    else:
                        logging.warning(
                            f"Could not resolve slot '{field_name}' for {cls_name}"
                        )

                instance = cls(**kwargs)
                body_id_to_instance[body_id] = instance
                instantiated.append(
                    {
                        "body_id": body_id,
                        "class": cls_name,
                        "status": "created",
                    }
                )
                logging.info(f"Created {cls_name} instance with slots filled")

            except Exception as e:
                logging.warning(f"Failed to create {cls_name} for {body_id}: {e}")
                instantiated.append(
                    {
                        "body_id": body_id,
                        "class": cls_name,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        # Save instantiation results
        instantiation_file = args.summary_json.parent / "instantiation_results.json"
        with open(instantiation_file, "w") as f:
            json.dump(instantiated, f, indent=2)
        logging.info(f"Instantiation results saved to {instantiation_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine class structure from VLM classification summary"
    )
    parser.add_argument(
        "summary_json",
        type=Path,
        help="Path to summary JSON file (body_id -> class mapping)",
    )
    parser.add_argument(
        "world_database_id",
        type=int,
        help="DB ID of the world stored in the previous step",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to output slot filling results (default: <summary_dir>/slot_filling_results.json)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.9,
        help="Minimum confidence threshold (default: 0.9)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate prompts but don't query VLM or instantiate objects",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Maximum number of candidate objects per slot from cheap LLM (default: 8)",
    )
    args = parser.parse_args()
    main(args)
