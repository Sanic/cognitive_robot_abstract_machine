# RoboKudo
RoboKudo is an open source framework for robot perception for ROS2.

## Installation instructions for Ubuntu (tested on 24.04)
The robokudo consists of two parts: The core `robokudo` python package and related ROS2 packages. 
A key ROS2 package is `robokudo_ros`, which provides entrypoint scripts to start RoboKudo or launch query-related tooling.

To install RoboKudo, first [install the main CRAM stack according to it's README](https://github.com/cram2/cognitive_robot_abstract_machine/?tab=readme-ov-file#installation).
Things to note:
* Use UV instead of poetry. Make sure to `workon cram-env` before calling uv.
* After dependencies are installed with uv,
    - Make sure that `source /opt/ros/jazzy/setup,bash` is in your .bashrc and 
    - Set up your workspace according to the `setup_ros_workspace.sh` script mentioned in the CRAM README.
* After that, follow the step where a symlink for the ROS2 packages is created in your ROS2 workspace

### PyCharm setup
- Install PyCharm 
- Open a terminal, source your ROS workspace and then start PyCharm from that terminal
- In PyCharm:
  - Open the folder where you've cloned the CRAM repository
  - Open the ROS Workspace folder as well and choose the 'Attach' option 
  - Set your `cram-env` virtualenv as the Project Interpreter. File → Settings → Python → Interpreter
  - In the left pane, open your ROS workspace, select `build`, `install` and `log`, right-click them and 'Mark directory as'  → 'Excluded'
  - In the left pane, open your ROS workspace and navigate to the `robokudo_ros` folder. Go to `robokudo_ros/scripts` and right-click `main.py` and choose 'Debug' to create your first run config.

#### Troubleshooting
- If your RoboKudo ROS packages cannot be resolved in PyCharm but the actual execution works, check the correct setup of the Project Dependencies. To do this, go to File → Settings → Project → Project Dependencies. Make sure that under `ros2_ws`, the checkbox next to `cognitive_robot_abstract_machine` is ticked. Also check the other way (`ros2_ws` in `cognitive_robot_abstract_machine` is ticked).

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
