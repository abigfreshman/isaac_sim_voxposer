import os
import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import rot_matrix_to_quat


class Move:
    """
        Controlling the posture and movement of the robot arm
    """
    def __init__(self) -> None:

        self.joint_position_cmd_list = []
        # gripper 状态：打开
        self.gripper_position_cmd = [0.04, 0.04]
        # franka 最初位姿
        self.joint_position_cmd = [0.04409, -0.17844, -0.03579,  -1.46597, -0.00767, 1.27802,  0.78540]
        
    def init_franka_pose(self, world):

        dc = _dynamic_control.acquire_dynamic_control_interface()
        art = dc.get_articulation("/World/franka/panda_link0")

        #  check dofs and joints informations of franka robot
        # num_joints = dc.get_articulation_joint_count(art)
        # num_dofs = dc.get_articulation_dof_count(art)
        # num_bodies = dc.get_articulation_body_count(art)
        # print(f"num_joints:{num_joints}")
        # print(f"num_dofs:{num_dofs}")
        # print(f"num_bodies:{num_bodies}")

        # init gripper pose
        gripper_dof_ptrs = [dc.find_articulation_dof(
            art, f"panda_finger_joint1"), dc.find_articulation_dof(art, f"panda_finger_joint2")]

        dc.wake_up_articulation(art)

        for idx, gripper_dof_ptr in enumerate(gripper_dof_ptrs):
            dc.set_dof_position_target(
                gripper_dof_ptr, self.gripper_position_cmd[idx])

        # init articulation poses
        dc = _dynamic_control.acquire_dynamic_control_interface()
        art = dc.get_articulation("/World/franka/panda_link0")

        dof_ptrs = [dc.find_articulation_dof(
            art, f"panda_joint{i}") for i in range(1, 8)]

        dc.wake_up_articulation(art)
        for _ in range(36):
            world.step(render=True)
        for idx, dof_ptr in enumerate(dof_ptrs):
            dc.set_dof_position_target(
                dof_ptr, self.joint_position_cmd[idx])
             
    def init_articulation(self):
        """
            init the articulation kinematics solver
        """
        self.articulation = Articulation(
            "/World/franka/panda_link0")
        if not self.articulation.handles_initialized:
            self.articulation.initialize()

        mg_extension_path = get_extension_path_from_name(
            "omni.isaac.motion_generation")

        kinematics_config_dir = os.path.join(
            mg_extension_path, "motion_policy_configs")

        kinematics_solver = LulaKinematicsSolver(                       # 初始化的运动学求解器，求解关节dofs
            robot_description_path=kinematics_config_dir +
            "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=kinematics_config_dir + "/franka/lula_franka_gen.urdf"
        )

        end_effector_name = "right_gripper"
        self.articulation_kinematics_solver = ArticulationKinematicsSolver(
        self.articulation, kinematics_solver, end_effector_name)

        robot_base_translation, robot_base_orientation = self.articulation.get_world_pose()
        kinematics_solver.set_robot_base_pose(
            robot_base_translation, robot_base_orientation)
        
    def move_to_by_apply_action(self, pose):
        """
            move robot arm to the next one pose using articulation <apply_action> function
        Args:
            pose: world pose of the end effector, 
        """
        print(pose)
        target_position = pose[:3]
        target_quat = pose[3:7]
        length = pose[-1]
        # 利用关节运动学求解器将终端执行器位姿转化为机械臂可直接执行的姿态
        action, success = self.articulation_kinematics_solver.compute_inverse_kinematics(
                target_position, target_quat) ########
        
        # print(f"{action}, 成功标志:{success}")
        print(f"转化成功标志:{success}")
    
        if success:
            joint_position_cmd = action.get_dict()["joint_positions"]
        else:
            print("转换失败, 跳过该节点")
            return
        # 指定抓手的位移，最大为【0.04， 0.04】表示gripper完全打开
        if pose[-2] == 1:
            joint_position_cmd.extend([0.04, 0.04])
        elif pose[-2] == 0:
            joint_position_cmd.extend([length/2, length/2])
        else:
            print(f"抓手状态错误,必须为0或1,得到{pose[-2]}")

        action = ArticulationAction(joint_positions=np.array(joint_position_cmd))
        # print(action)
        self.articulation.apply_action(action)

    def move_to_by_dc(self, pose):
        # print(f"输入位置：{pose[:3]}: 输入旋转：{pose[3:7]}")
        target_position = pose[:3]
        target_quat = pose[3:7]
        length = pose[-1]
        # 求逆解， 由终端执行器到关节自由度
        action, success = self.articulation_kinematics_solver.compute_inverse_kinematics(
                target_position, target_quat)
        
        if success:
            joint_position_cmd = action.get_dict()["joint_positions"]
        else:
            print("转换失败, 跳过该节点")
            return
        # print(f"转化后的dofs位移:{joint_position_cmd}")
        # 获取机械臂控制接口
        dc = _dynamic_control.acquire_dynamic_control_interface()
        art = dc.get_articulation("/World/franka/panda_link0")
        # 获取franka机械臂7个关节dofs
        dof_ptrs = [dc.find_articulation_dof(                      # retun ->int
            art, f"panda_joint{i}") for i in range(1, 8)]

        dc.wake_up_articulation(art)
        # 指定每个关节具体dof值， 运动到目标pose
        for idx, dof_ptr in enumerate(dof_ptrs):
            dc.set_dof_position_target(
                dof_ptr, joint_position_cmd[idx])
        # print(f"关节姿态：{joint_position_cmd}")
        # gripper action
        if pose[-2] == 1:
            gripper_position_cmd = np.array([0.04, 0.04])
            # print("打开抓手")
        elif pose[-2] == 0:
            gripper_position_cmd = np.array([length/2, length/2])
            # print("闭合抓手")
        else:
            print(f"抓手状态错误,必须为0或1,得到{pose[-2]}")
        dc = _dynamic_control.acquire_dynamic_control_interface()
        art = dc.get_articulation("/World/franka/panda_link0")

        gripper_dof_ptrs = [dc.find_articulation_dof(
            art, f"panda_finger_joint1"), dc.find_articulation_dof(art, f"panda_finger_joint2")]

        dc.wake_up_articulation(art)
        # 执行gripper动作
        for idx, gripper_dof_ptr in enumerate(gripper_dof_ptrs):
            dc.set_dof_position_target(
                gripper_dof_ptr, gripper_position_cmd[idx])
        # 求正解：根据当前帧关节pose计算得到终端执行器pose， 判断动作是否执行完成
        pose_from_joints_action = self.articulation_kinematics_solver.compute_end_effector_pose()
        position, orientition = pose_from_joints_action[0], pose_from_joints_action[1]
        quat = rot_matrix_to_quat(orientition)
        # print(f"输出位置：{position}, 输出旋转：{quat}")
        # input()
        return (position, quat)

    def compute_end_effector_pose_dy_articulation_pose(self):
        """
        根据当前帧关节pose计算得到终端执行器pose
        Returns:
            position: end effector position
            quat: end effector orientation
        """
        pose_from_joints_action = self.articulation_kinematics_solver.compute_end_effector_pose()
        position, orientition = pose_from_joints_action[0], pose_from_joints_action[1]
        quat = rot_matrix_to_quat(orientition)

        return position, quat