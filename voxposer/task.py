import numpy as np
from utils import bcolors

from omni.isaac.core import World
import omni.replicator.core as rep
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units

from omni.isaac.franka import Franka
from fakerealsense import FakeRealSense
from omni.isaac.core.objects import DynamicCuboid
import logging
import omni.kit.viewport.utility as vp_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PutRubbishInBin(BaseTask):
    """
    create task scene and scene objects
    Args:
        name: str: the short name of task 
    """
    def __init__(self,
                 world,
                 name: str = "put rubbish in bin",
                 bin_initial_position = None,
                 bin_initial_orientation = None,
                 bin_size = None,
                 rubbish_inital_position = None,
                 rubbish_initial_orientation=None,
                 rubbish_size = None,
                 first_obstruction_pos=None,
                 second_obstruction_pos=None,
                 obstruction_ori=None,
                 obstruction_size=None,
                 offset= 0.0
                 ) -> None:
        super().__init__(name=name, offset=None)

        # self.name = name
        self.franka_robot = None
        self.bin = None
        self.rubbish = None
        self.first_obstruction = None
        self.second_obstruction = None

        self.bin_initial_position  = bin_initial_position
        self.bin_initial_orientation = bin_initial_orientation
        self.bin_size = bin_size

        self.rubbish_initial_position = rubbish_inital_position
        self.rubbish_initial_orientation = rubbish_initial_orientation
        self.rubbish_size = rubbish_size

        self.first_obstruction_pos = first_obstruction_pos
        self.second_obstruction_pos = second_obstruction_pos
        self.obstruction_ori = obstruction_ori
        self.obstruction_size = obstruction_size

        self.offset = offset

        self.lookat_vector = [0,0,1]
        
        #  init the scene objects positions and orientation
        if self.bin_size == None:
            self.bin_size = np.array([0.15, 0.25, 0.15])/get_stage_units()
        
        if self.rubbish_initial_position == None:
            self.rubbish_initial_position = np.array([0.5, -0.5, 0])/get_stage_units()
        if self.rubbish_size == None:
            self.rubbish_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()

        if self.first_obstruction_pos == None:
            self.first_obstruction_pos = np.array([self.rubbish_initial_position[0]+5*self.rubbish_size[0], 
                                                  self.rubbish_initial_position[1]+5*self.rubbish_size[0], 0]
                                                 )
        if self.second_obstruction_pos ==None:
            self.second_obstruction_pos = np.array([self.rubbish_initial_position[0]-5*self.rubbish_size[0], 
                                                  self.rubbish_initial_position[1]-5*self.rubbish_size[0], 0]
                                                 )         
        if self.obstruction_size == None:
            self.obstruction_size = self.rubbish_size

        if self.bin_initial_position ==  None:
            self.bin_initial_position = np.array([0.5, 0.5, 0]) / get_stage_units()
        
        self.bin_initial_position = self.bin_initial_position + self.offset

        self.work_space= [1, 2, 1.5]

        self.object_positon = {"bin":self.bin_initial_position, 
                               "rubbish":self.rubbish_initial_position,
                               "first_obstruction":self.first_obstruction_pos,
                               "second_obstruction":self.second_obstruction_pos
                               }
        self.object_size = {"bin":self.bin_size, 
                            "rubbish": self.rubbish_size, 
                            "first_obstruction":self.obstruction_size,
                            "second_obstruction":self.obstruction_size
                            }
        
        world.scene.add_default_ground_plane()
        # add_reference_to_stage(usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Environments/Grid/gridroom_curved.usd", prim_path="/GridRoom")
 
        create_prim("/World/cube")
        # create_prim("/World/cube/rubbish_cube")
        # create_prim("/World/cube/tomato1")
        # create_prim("/World/cube/tomato2")

        self.rubbish = world.scene.add(DynamicCuboid(
            name='rubbish',
            position=self.rubbish_initial_position,
            orientation=self.rubbish_initial_orientation,
            prim_path='/World/cube/rubbish_cube',
            scale=self.rubbish_size,
            size=1.0,
            color=np.array([0, 1, 0])
        ))
 
        self.first_obstruction = world.scene.add(DynamicCuboid(
            name='tomato1',
            position=self.first_obstruction_pos,
            orientation= self.obstruction_ori,
            prim_path='/World/cube/tomato1',
            scale=self.obstruction_size,
            size=1.0,
            color=np.array([0, 0, 0.8])
        ))

        self.second_obstruction = world.scene.add(DynamicCuboid(
            name='tomato2',
            position=self.second_obstruction_pos,
            orientation= self.obstruction_ori,
            prim_path='/World/cube/tomato2',
            scale=self.obstruction_size,
            size=1.0,
            color=np.array([0, 0, 0.8])
        ))

        #  add the scene usd file like rubbish and bin
        bin_usd_path = "/home/ps/isaacsim42/isaac_sim_voxposer/scene_obj_usd/bin_no_reference.usd"
        bin_prim_path = "/World/Steel_bin"
        room_usd_path = "/home/ps/isaacsim42/isaac_sim_voxposer/scene_obj_usd/sample_room_move.usd"
        room_prim_path = "/World/Room"
        # rubbish_usd_path = "/home/ps/Desktop/usd_change/rubbish.usd"
        # rubbish_prim_path = "/World"
        add_reference_to_stage(usd_path=bin_usd_path, 
                               prim_path=bin_prim_path)
        # add_reference_to_stage(usd_path=rubbish_usd_path, 
        #                        prim_path=rubbish_prim_path)
        add_reference_to_stage(usd_path=room_usd_path, 
                               prim_path=room_prim_path)

        world.scene.add(XFormPrim(prim_path="/World/Steel_bin", name="bin"))
        # world.scene.add(XFormPrim(prim_path="/World/Paper_Ball", name="rubbish"))
        world.scene.add(XFormPrim(prim_path="/World/Room", name="sample_room"))

        self.bin = XFormPrim("/World/Steel_bin")
        self.bin.set_world_pose(position=[-0.6, 0, 3.83])
        # self.rubbish = XFormPrim("/World/Paper_Ball")
        self.room = XFormPrim("/World/Room")
        self.room.set_world_pose(position=[0,0,-0.6])
        self.cube = XFormPrim("/World/cube")
        self.cube.set_world_pose(position=[-0.6,0,4.5])

        # add franka robot
        self.franka_robot = world.scene.add(Franka(
            name='franka',
            prim_path='/World/franka',
            position=[-0.5, 0, 3.9],
            orientation=[1.,0.,0.,0.]
        ))

        # attach sementic property to objects to read segmention mask 
        rep.modify.semantics([("class2", "bin")], ["/World/Steel_bin"])
        # rep.modify.semantics([("class3", "rubbish")], ["/World/Paper_Ball"])
        rep.modify.semantics([("class3", "rubbish")], ['/World/cube/rubbish_cube'])
        rep.modify.semantics([("class4", "franka")], ["/World/franka"])
        rep.modify.semantics([("class5", "tomato1")], ["/World/first_obs_cube"])
        rep.modify.semantics([("class6", "tomato2")], ["/World/second_obs_cube"])

        self._task_objects[self.bin.name] = self.bin
        self._task_objects[self.rubbish.name] = self.rubbish
        self._task_objects[self.first_obstruction.name] = self.first_obstruction
        self._task_objects[self.second_obstruction.name] = self.second_obstruction
        self._task_objects[self.franka_robot.name] = self.franka_robot
        # 创建相机
        self.cameras = {}
 
        self.camera_names = ["front", 'left_shoulder', "rigth_shoulder", "overhead", "wrist"]
        print("#"*100)
        print(f"{bcolors.OKBLUE}start to create cameras")
        # 创建相机之前先创建路径
        create_prim("/World/cameras")
        create_prim("/World/cameras/front")
        create_prim("/World/cameras/left_shoulder")
        create_prim("/World/cameras/right_shoulder")
        create_prim("/World/cameras/overhead")

        # cameras positon with resolution (640, 480), change cameras positoion with different resolution to get better imaging results
        # self.front_cam = FakeRealSense(name=self.camera_names[0], 
        #                                position=[3.0, -0.0, 2.0], 
        #                                orientation=[-0.35355, 0.61237, 0.61237, -0.35355],
        #                                base_link='/World/front')
        
        # self.left_shoulder_cam = FakeRealSense(name=self.camera_names[1], 
        #                                        position=[0.3, 1.5, 1.8],
        #                                        orientation=[0.2706, 0.65328, -0.65328, 0.2706],
        #                                        base_link="/World/left_shoulder")
        # self.right_shoulder_cam = FakeRealSense(name=self.camera_names[2],
        #                                         position=[0.3, -1.5, 1.8],
        #                                         orientation=[0.2706, -0.65328, 0.65328, 0.2706],
        #                                         base_link="/World/right_shoulder")
        # self.overhead_cam = FakeRealSense(name=self.camera_names[3], 
        #                                   position=[0.3, -0.0, 3.0],
        #                                   orientation=[0.0, 0.70711, -0.70711, 0.0],
        #                                   base_link="/World/overhead")
        # self.wrist_cam = FakeRealSense(position=[0.0, -0.0, 0.05], 
        #                                orientation=[1., 0., 0., 0.],
        #                                base_link="/World/franka/panda_hand"
        #                                )
        
        # self.cameras_pos = [[3.0, -0.0, 2.0],
        #                     [0.3, 1.5, 1.8],
        #                     [0.3, -1.5, 1.8],
        #                     [0.3, -0.0, 3.0],
        #                     [0., 0., 0.05]]

        # cameras position with resolution (1024, 798)
        self.front_cam = FakeRealSense(name=self.camera_names[0], 
                                       position=[2.0, -0.0, 1.5], 
                                       orientation=[-0.2706, 0.65328, 0.65328, -0.2706],
                                       base_link='/World/cameras/front')
        
        self.left_shoulder_cam = FakeRealSense(name=self.camera_names[1], 
                                               position=[0.3, 1.0, 1.5],
                                               orientation=[0.21263, 0.67438, -0.67438, 0.21263],
                                               base_link="/World/cameras/left_shoulder")
        self.right_shoulder_cam = FakeRealSense(name=self.camera_names[2],
                                                position=[0.3, -1., 1.5],
                                                orientation=[0.21263, -0.67438, 0.67438, 0.21263],
                                                base_link="/World/cameras/right_shoulder")
        self.overhead_cam = FakeRealSense(name=self.camera_names[3], 
                                          position=[0.3, -0.0, 2.0],
                                          orientation=[0.0, 0.70711, -0.70711, 0.0],
                                          base_link="/World/cameras/overhead")
        self.wrist_cam = FakeRealSense(position=[0.0, -0.0, 0.05], 
                                       orientation=[1., 0., 0., 0.],
                                       base_link="/World/franka/panda_hand"
                                       )
        
        self.cameras_pos = [[2., -0.0, 1.5],
                            [0.3, 1., 1.5],
                            [0.3, -1., 1.5],
                            [0.3, -0.0, 2.],
                            [0., 0., 0.05]]

        self.cameras[self.camera_names[0]] = self.front_cam
        self.cameras[self.camera_names[1]] = self.left_shoulder_cam
        self.cameras[self.camera_names[2]] = self.right_shoulder_cam
        self.cameras[self.camera_names[3]] = self.overhead_cam
        self.cameras[self.camera_names[4]] = self.wrist_cam

        self.cameras_root = XFormPrim("/World/cameras")
        self.cameras_root.set_world_pose(position=[0,0,5])
        
        print('相机创建完成')
        # vp_utils.get_active_viewport().set_active_camera("/World/cameras/front")
        # from omni.isaac.core.utils.prims import delete_prim

        delete_prim("/World/Room/turtlebot_tutorial/ActionGraph_drive")


    def get_cam(self):
        return self.cameras 

    def get_observations(self) -> dict:
        """
        Returns current observations from the objects needed for the behavioral layer.
        Returns:
            dict: [description]
        """
        # joints_state = self.franka_robot.get_joints_state()
        bin_position, bin_orientation = self.bin.get_world_pose()
        end_effector_position, end_effector_orientation = self.franka_robot.gripper.get_world_pose()
        _, gripper_local_orientation = self.franka_robot.gripper.get_local_pose()
        rubbish_position, rubbish_orientation = self.rubbish.get_world_pose()
        obstruction_pos_0, obstruction_ori_0= self.first_obstruction.get_world_pose()
        obstruction_pos_1, obstruction_ori_1= self.second_obstruction.get_world_pose()
        # TODO: change values with USD
        return {
            "bin": {
                "position": bin_position,
                "orientation": bin_orientation,
                "target_position": self.bin_initial_position,
                "size": self.bin_size,
            },
            "franka": {
                # "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
                "end_effector_orientation": end_effector_orientation,
                "gripper_local_orientation": gripper_local_orientation,
            },
            "rubbish":{"position":rubbish_position, 
                       "orientation":rubbish_orientation, 
                       "target_position":self.bin_initial_position, 
                       "size":self.rubbish_size
            },
            "tomato1":{"position":obstruction_pos_0,
                                 "orientation":obstruction_ori_0,
                                 "size":self.obstruction_size

            },
            "toamto2":{"position":obstruction_pos_1,
                                 "orientation":obstruction_ori_1,
                                 "size":self.obstruction_size

            }
        }