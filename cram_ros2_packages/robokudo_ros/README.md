# robokudo_ros

This is a ROS wrapper for RoboKudo.

Please finish the installation of the CRAM architecture based on the README
in: https://github.com/cram2/cognitive_robot_abstract_machine/tree/robokudo/ first, before installing this wrapper.
While doing so make sure to:

- Use `python3 -m colcon build ...` instead of the `colcon build ...` while the CRAM virtual environment is active
  to build the ROS workspace, so that the python interpreter of the virtual environment is used instead of the system
  interpreter. This makes sure that all packages from the virtual environment are actually available to the ROS scripts.
- Install `setuptools<80` with `pip install "setuptools<80"` when using the `--symlink-install` option due to colcon
  incompatibilities with newer versions.

## Installation instructions for Ubuntu (tested on 24.04)

TBD

### Tutorials

https://robokudo.ai.uni-bremen.de/

### How to cite

```
@inproceedings{mania2024robokudo,
	title={An Open and Flexible Robot Perception Framework for Mobile Manipulation Tasks},
	author={Mania, Patrick and Stelter, Simon and Kazhoyan, Gayane and Beetz, Michael},
	booktitle={2024 International Conference on Robotics and Automation (ICRA)},
	year={2024},
	url={https://ai.uni-bremen.de/papers/mania2024robokudo.pdf},
	note={},
	organization={IEEE}
}
```
