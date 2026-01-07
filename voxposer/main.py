import openai

from arguments import get_config
from isaac_env import VoxposerIsaccEnv
from interfaces import setup_LMP
from utils import set_lmp_objects
from visualizers import ValueMapVisualizer

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from task import PutRubbishInBin
from move import Move

from omni.isaac.core import World
from isaac_sim_voxposer.utils import setup_logger
import argparse


openai.api_key = "sk-l2TuLVvHjIrRe9XZh7J79jnncxZQVpDPtpXVYXfCwbUEaESj"
openai.api_base= "https://api.chatanywhere.tech/v1"


def main(args):
    world = World()

    logger.info("Starting initialization task")
    my_task = PutRubbishInBin(world, name=args.task_name)
    world.add_task(my_task)

    logger.info("Task initialization completed")

    config = get_config(args.config)
    logger.info("Configuration file loaded successfully")

    if not args.visualize:
        visualizer = None
    else:
        visualizer = ValueMapVisualizer(config["visualizer"])

    logger.info("Starting environment initialization")

    world.reset()

    action = Move()
    action.init_franka_pose(world)
    action.init_articulation()
    env = VoxposerIsaccEnv(my_task, action, world, simulation_app, visualizer=visualizer)
    robot_name = my_task.franka_robot.name

    logger.info("Environment initialization completed successfully")

    logger.info("Adding sub-task LMPs")
    lmps, lmp_env = setup_LMP(env, config, world, debug=False)
    logger.info("LMPs added successfully")
    voxposer_ui = lmps['plan_ui']

    scene_obj = ["bin", "rubbish", "tomato1", "tomato2"]

    env.task_scene_objects = scene_obj
    env.robot_name = robot_name
    env.grasped_obj_name = ["rubbish"]
    set_lmp_objects(lmps, scene_obj)

    logger.info("Scene objects setup completed")
    logger.info("Starting LMP action execution...")
    env.reset()
    # while True:
    #     simulation_app.update()
    # input()
    # print("enter to continue ....")
    voxposer_ui(args.description)

    while True:
        simulation_app.update()

    # simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="isaac-sim", help="Path to the config file")
    parser.add_argument("--visualize", action="store_true", help="Wether to visualize and save the planned path.")
    parser.add_argument("--task-name", default="put_rubbish_in_bin", help="Name of the task")
    parser.add_argument("--description", default="put the rubbish into the bin", help="Description of the task")

    args = parser.parse_args()
    
    logger = setup_logger("main")
    main(args)