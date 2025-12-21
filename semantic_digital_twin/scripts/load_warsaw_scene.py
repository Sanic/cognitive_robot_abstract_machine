from pathlib import Path

from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader

if __name__ == "__main__":
    obj_dir = Path("/home/ben/devel/iai/src/cognitive_robot_abstract_machine/Objects")
    warsaw_world_loader = WarsawWorldLoader(obj_dir)
    warsaw_world_loader.export_semantic_annotation_inheritance_structure(
        output_directory=Path("../resources/warsaw_data/json_exports/")
    )
    warsaw_world_loader.export_scene_to_pngs(
        number_of_bodies=4,
        output_directory=Path("../resources/warsaw_data/scene_images/"),
    )
