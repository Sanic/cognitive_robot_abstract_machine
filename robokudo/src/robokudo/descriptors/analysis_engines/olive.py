from __future__ import annotations

# %% imports

from dataclasses import dataclass, field

from robokudo.descriptors.analysis_engines import foundation_stereo
from robokudo.descriptors.camera_configs.config_olive import OliveCameraConfig

# %% sensor display


@dataclass
class AnalysisEngine(foundation_stereo.AnalysisEngine):
    """
    Display Olive rectified RGB-D sensor data and its reconstructed cloud.
    """

    camera_config: OliveCameraConfig = field(default_factory=OliveCameraConfig)
    """
    Camera topics, optical frame and synchronization settings.
    """

    def name(self) -> str:
        return self.camera_config.registry_name
