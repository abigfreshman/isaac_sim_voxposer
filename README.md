# Setup Instructions
This code needs to be run in the NVIDIA simulation platform isaac sim. Please refer to the [official documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html) to 
download and install isaac sim. Run the Python script in standalone mode. In the demo, we use isaac sim 2023.1.1 <br>
#### Create a conda environment:<br>  
```conda create -n voxposer-env python=3.9```<br>
```conda activate voxposer-env```<br>
#### Install other dependencies:<br>
```pip install -r requirements.txt```<br>

# Running Demo  <br>
cd to the isaac sim file and run`./python.sh ./path/to/the/main.py` to display the demo. You can see [Hellow World](https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_hello_world.html)
to konw more about how to run a python script in isaac.<br>

# Code Structure <br>

#### Core to VoxPoser:<br>
```main.py```: Run for this demo<br>
```LMP.py```: Implementation of Language Model Programs (LMPs) that recursively generates code to decompose instructions and compose value maps for each sub-task.<br>
```interfaces.py```: Interface that provides necessary APIs for language models (i.e., LMPs) to operate in voxel space and to invoke motion planner.<br>
```planners```: Implementation of a greedy planner that plans a trajectory (represented as a series of waypoints) for an entity/movable given a value map.<br>
```controllers.py```: Given a waypoint for an entity/movable, the controller applies (a series of) robot actions to achieve the waypoint.<br>
```dynamics_models.py```: Environment dynamics model for the case where entity/movable is an object or object part. This is used in controllers.py to perform MPC.<br>
```prompts/isaac_sim```: Prompts used by the different Language Model Programs (LMPs) in VoxPoser.<br>
#### Environment and utilities:<br>

```isaac_env.py```: Wrapper of isaac env to expose useful functions for VoxPoser.<br>
```configs/rlbench_config.yaml```: Config file for all the involved modules in RLBench environment.<br>
```arguments.py```: Argument parser for the config file.<br>
```LLM_cache.py```: Caching of language model outputs that writes to disk to save cost and time.<br>
```utils.py```: Utility functions.<br>
```visualizers.py```: A Plotly-based visualizer for value maps and planned trajectories.<br>
```task.py```: Task excuted in the demo display. The task is base on isaac sim scene.<br>
```move.py```: Dynamic control of the franka panda robot. Move the robot arm to the target pose<br>
```fakerrealscene.py```: Create faker RealScene cameras in isaac sim world and read depth and points data.<br>

### Others folder:
```cache```: Preserves LMP decomposing and planning information. Saves the disassembly and planning information of LMP. You can call this cache information in the subsequent execution process to speed up the movement process.<br>
```visualizer```: Visualization files of planning paths and scene point clouds.<br>
```scene_boject_usd```： Objects USD file used in the task scnen build.<br>

# Note:
1. In isaac sim, all task scenes need to be built manually, so after changing the task, you need to re-set the scene objects in ```main.py```, including interesting object or the robot.<br>
2. More complex tasks mean more waypoints to plan. For different tasks, you can adjust the value map size in ```isaac_config.yaml``` and reduce or increase the search range of
waypoints in ```interfaces.py``` during path planning to ensure smoother movement. Note that there is a trade-off between the number of path points and the movement time <br>

