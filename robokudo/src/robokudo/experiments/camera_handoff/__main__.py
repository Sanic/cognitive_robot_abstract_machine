# %% imports

"""
Compare head-camera initialization with monocular approach feedback.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from robokudo.experiments.camera_handoff.experiment import CameraHandoffExperiment
from robokudo.experiments.camera_handoff.textured_experiment import (
    TexturedHandoffExperiment,
)
from robokudo.experiments.camera_handoff.textured_models import (
    ModelAsset,
    TexturedModel,
)

# %% command line


class CommandOption(StrEnum):
    """
    Command-line options for the camera handoff experiment.
    """

    OUTPUT = "--output"
    MODEL = "--model"


@dataclass
class ExperimentCommand:
    """
    Run a camera handoff comparison and report its recording directory.
    """

    output_root: Path = Path("/tmp/robokudo-camera-handoff")
    """
    Parent directory for timestamped default recordings.
    """

    def run(self) -> None:
        """
        Read the output destination and execute the default experiment.
        """
        parser = ArgumentParser(description=__doc__)
        parser.add_argument(
            CommandOption.OUTPUT,
            type=Path,
            default=self.output_root / datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        )
        parser.add_argument(CommandOption.MODEL, choices=list(ModelAsset), default=None)
        arguments = parser.parse_args()
        experiment = (
            CameraHandoffExperiment()
            if arguments.model is None
            else TexturedHandoffExperiment(
                model=TexturedModel(ModelAsset(arguments.model))
            )
        )
        result = experiment.run(arguments.output)
        print(f"Images and measurements: {arguments.output.resolve()}")
        print(
            f"Final lateral error: head-only {result.head_only.final_lateral_error:.4f} m; feedback {result.feedback.final_lateral_error:.4f} m"
        )
        if not result.feedback.completed:
            parser.exit(
                status=1,
                message="Feedback approach stopped before successful alignment.\n",
            )


if __name__ == "__main__":
    ExperimentCommand().run()
