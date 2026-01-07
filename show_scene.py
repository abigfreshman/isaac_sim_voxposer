import openai

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from env.task import PutRubbishInBin
from omni.isaac.core import World
from utils import setup_logger
import argparse
import time


openai.api_key = "sk-l2TuLVvHjIrRe9XZh7J79jnncxZQVpDPtpXVYXfCwbUEaESj"
openai.api_base= "https://api.chatanywhere.tech/v1"


def main(args):
    world = World()

    logger.info("Starting initialization task")
    my_task = PutRubbishInBin(world, name=args.task_name)

    while True:
        simulation_app.update()
        time.sleep(0.01)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="isaac-sim", help="Path to the config file")
    parser.add_argument("--visualize", action="store_true", help="Wether to visualize and save the planned path.")
    parser.add_argument("--task-name", default="put_rubbish_in_bin", help="Name of the task")
    parser.add_argument("--description", default="put the rubbish into the bin", help="Description of the task")

    args = parser.parse_args()
    
    logger = setup_logger("main")
    main(args)