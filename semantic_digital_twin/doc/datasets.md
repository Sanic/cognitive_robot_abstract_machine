# Datasets

Semantic Digital Twin can load datasets from internet resources.
The results of the loaded datasets are completely function digital twins 
(World instances including Semantic Annotations, Kinematics, etc.).


## Sage
Scenes from [Sage](https://nvlabs.github.io/sage/) can be loaded with:

```python
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader

loader = Sage10kDatasetLoader()
scene = loader.create_scene(scene_url=Sage10kDatasetLoader.available_scenes()[0])
world = scene.create_world()
```

## Sapien / PartNet

Articulated assets from the [PartNet-Mobility](https://sapien.ucsd.edu/browse) dataset can be loaded with:

```python
from semantic_digital_twin.adapters.partnet_mobility_dataset.loader import PartNetMobilityDatasetLoader

loader = PartNetMobilityDatasetLoader()
world = loader.load(model_id=179) # model_id can be found at https://sapien.ucsd.edu/browse
```

Note that this requires the `sapien` library to be installed and the `SAPIEN_ACCESS_TOKEN` environment variable to be set.

## RoboCasa

Objects, fixtures, and full kitchen scenes from [RoboCasa](https://github.com/robocasa/robocasa) can be loaded with:

```python
from semantic_digital_twin.adapters.robocasa_dataset.loader import RoboCasaDatasetLoader
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
    RoboCasaObjectCategory,
)

loader = RoboCasaDatasetLoader()

kitchen_world = loader.load_kitchen(layout_id=..., style_id=...)  # a full kitchen scene
appliance_world = loader.load_kitchen_appliance(RoboCasaKitchenApplianceCategory.CABINET)  # a single appliance
object_world = loader.load_object(RoboCasaObjectCategory.APPLE)  # a single object
```

Note that this requires the `robocasa` and `robosuite` libraries to be installed (`robosuite` must be
installed from git, `pip install git+https://github.com/ARISE-Initiative/robosuite.git`), and the
fixture/object assets to be downloaded via `python -m robocasa.scripts.download_kitchen_assets`
(pointed at by `RoboCasaDatasetLoader.directory`, `~/robocasa-assets` by default).
