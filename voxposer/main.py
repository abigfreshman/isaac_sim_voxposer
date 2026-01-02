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


openai.api_key = "sk-l2TuLVvHjIrRe9XZh7J79jnncxZQVpDPtpXVYXfCwbUEaESj"
openai.api_base= "https://api.chatanywhere.tech/v1"

print("接口正常，开始执行")
world = World()

print("#"*100)
print("开始初始化任务")
my_task = PutRubbishInBin(world, name="put_rubbish_in_bin")
world.add_task(my_task)

print("任务初始化完成")
print("#"*100)

config = get_config('isaac-sim')
print("配置文件加载成功")
print("#"*100)
# visualizer = ValueMapVisualizer(config["visualizer"])   # 可视化显示规划的路径，开启时额外增加运动时长, 默认为None，即关闭
visualizer = None           # 需要显示路径时反注释上一行
print("开始初始化环境")

world.reset()

action = Move()
action.init_franka_pose(world)
action.init_articulation()
env = VoxposerIsaccEnv(my_task, action, world, simulation_app, visualizer=visualizer)
robot_name = my_task.franka_robot.name

print("环境初始化成功")
print("#"*100)

print("添加各子任务LMP")
lmps, lmp_env = setup_LMP(env, config, world, debug=False)
print("LMP添加完成")
voxposer_ui = lmps['plan_ui']

description = "put the rubbish into the bin"
scene_obj = ["bin", "rubbish", "tomato1", "tomato2"]

env.task_scene_objects = scene_obj
env.robot_name = robot_name
env.grasped_obj_name = ["rubbish"]
set_lmp_objects(lmps, scene_obj)

print("完成场景物品设置")
print("#"*100)
print("开始调用LMP执行动作")
env.reset()
voxposer_ui(description)

while True:
    simulation_app.update()

simulation_app.close()