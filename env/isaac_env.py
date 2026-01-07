import numpy as np
import json
import open3d as o3d

from isaac_sim_voxposer.utils import normalize_vector, bcolors
import cv2
import ast
import open3d as o3d 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoxposerIsaccEnv:
    def __init__(self, task, action,  world, simulation_app, visualizer=None) -> None:
        """
        task对应的环境类, 实现task scene中的场景状态检测和物品信息获取
        Args:
            task: initialized task instance,
            action: initialized Move instance to move the robot arm
            world: scene world
            simulation_app: this app to make sure get current frame
            visualizer: Visualize value map and plan path
        """
        
        self.visualizer = visualizer
        self.task = task
        # self.gripper_to_articluation = gripper_to_articulation
        # self.action = Move()
        self.action = action
        self.my_world = world
        self.simulation_app = simulation_app
        self.visualizer = visualizer

        self.init_length = np.array([0.04, 0.04])

        self.workspace_min = np.array([-0.5, -1., 3.5])
        self.workspace_max = np.array([0.5, 0.5, 5])

        self.workspace_bounds_min = self.workspace_min
        self.workspace_bounds_max = self.workspace_max

        self.work_space = self.workspace_max - self.workspace_min  # 定义工作空间大小：m

        self.cameras_name = ["front", 'left_shoulder', "rigth_shoulder", "overhead", "wrist"]
       

        self.cameras = self.task.get_cam()
        self.observation = self.task.get_observations()

        logger.info("相机正常加载")
       
        self.cam_extrinsic = {}
        self.lookat = np.array([0,0,1])
        self.lookat_vector = {}
        for name, cam in self.cameras.items():
            ext = cam.get_camera_extrinsic()
            self.cam_extrinsic[name] = ext
            lookat_vector = ext[:3, :3] @ self.lookat
            self.lookat_vector[name] = normalize_vector(lookat_vector)

        logger.info("相机参数读取正常")
        
        self.task_scene_objects = ['bin', "rubbish", "tomato1", "tomato2"]
        self.robot_name = "franka"
        self._reset_task_variables()
        self.grasped_obj_name = ["rubbish"]
        #  init franka and gripper pose
        self.action.init_franka_pose(self.my_world)
        self.action.init_articulation()

        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_min, self.workspace_max)

    def get_3d_obs_by_name(self, name):
        """
        get the corresponding points cloud and normal vector by name, 
        and only return the corresponding information of one object at a time

        Args:
            name (_type_): _description_

        Returns:
            _type_: object world points
        """
        assert name in self.task_scene_objects, f"Scene do not exit {name} object."
        points = []
        normal = []

        for cam_name, cam in self.cameras.items():
            while True:
                self.simulation_app.update()       # 保证读取到有效的当前帧图像和深度数据
                img_rgb, img_depth = cam.read()
                if img_depth is not None and np.isinf(img_depth).all() ==0  and img_rgb.any() !=0:
                    break
            world_points = np.array(cam.get_world_points()).reshape(-1, 3)

            # 点云可视化
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(world_points)
            # o3d.visualization.draw_geometries([pcd])
            
            label_idx = None
            mask = cam.get_instance_segmentation()

            # 查看 mask 图像
            # img = mask["data"]        #数据类型为uint32 
            # img = img.astype(np.uint8)
            # cv2.imshow("my_img", img)
            # cv2.waitKey(0)  
            # cv2.destroyAllWindows()

            
            # find the index of the input "name" object
            idToLabels = mask["info"]["idToSemantics"]
            for idx, label in idToLabels.items():
                for k, v in label.items():
                    if v == name:
                        label_idx = ast.literal_eval(idx)
                        # print(f"RGBA:{name}:{label_idx}, {type(label_idx)}")
            
            if label_idx == None:
                continue
            # create the world points with shape(1024, 798), but the read mask and depth has shape(798, 1024)
            data = np.transpose(mask["data"])
            height, width = data.shape

            # (BGRA)->uint32
            b,g,r,a = int(label_idx[0]), int(label_idx[1]), int(label_idx[2]), int(label_idx[3])
            idx = (a << 24) | (r << 16) | (g << 8) | b         

            # only get the interesting object mask 
            obj_mask = np.zeros_like(data, dtype=bool)
            # the obj_mask shape must keep with points shape
            for x in range(height):
                for y in range(width):
                    if data[x][y] == idx:
                        obj_mask[x][y] = True
            obj_mask = obj_mask.flatten()

            clip_points = get_clip_points(world_points, self.workspace_min, self.workspace_max)# workstation 大小剪切点云
            clip_mask = get_clip_points(world_points, self.workspace_min, self.workspace_max, obj_mask)
            cam_obj_points = np.array(clip_points[clip_mask]).reshape(-1, 3)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cam_obj_points)
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, self.lookat_vector[cam_name]) > 0
            cam_normals[flip_indices] *= -1
            normal.append(cam_normals)
            points.append(cam_obj_points)
        if len(points):
            points = np.concatenate(points, axis=0)
            normals = np.concatenate(normal, axis=0)
        else:
            while True:
                # self.simulation_app.update()
                raise ValueError(f"Cannot find the object named {name} in the scene. run show_scene to check")

        # 点云采样
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)

        logger.info(f"检测到<{name}>点云shape: {obj_points.shape}")
        logger.info(f"检测到<{name}>法向量shape: {obj_normals.shape}")
        logger.info(f"检测到<{name}>最终位置: {np.mean(obj_points, axis=0)}")

        return obj_points, obj_normals

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        get world points in the workstation, you also can choose to ignore the robot and grasped object points
        Returns:
            _type_: world points
        """
        points = []
        colors = []
        masks = []
        
        for cam_name, cam in self.cameras.items():
            img_rgb, img_depth = cam.read()
            img_rgb = img_rgb.reshape(-1,3)
            world_points = cam.get_world_points().reshape(-1, 3)
            mask = cam.get_instance_segmentation()    # shape(798, 1024)
            mask_data, mask_info = mask["data"], mask["info"]
            points.append(np.asarray(world_points))
            colors.append(np.asarray(img_rgb))
            masks.append(np.transpose(mask_data).flatten())

        total_points = np.concatenate(points, axis=0)
        total_colors = np.concatenate(colors, axis=0)
        total_masks = np.concatenate(masks, axis=0)

        # 利用workstation大小分割点云，过滤不需要的点云数据
        points = np.array(get_clip_points(total_points, self.workspace_min, self.workspace_max)).reshape(-1, 3) # 全场景点云
        colors = np.array(get_clip_points(total_points, self.workspace_min, self.workspace_max, img=total_colors)).reshape(-1, 3)
        masks = np.array(get_clip_points(total_points, self.workspace_min, self.workspace_max,img = total_masks ))

        if ignore_robot:
            # 忽略franka对应的点云，mask获取方式不变
            robot_idx = None          
            robot_name = self.robot_name
            idToLabels = mask_info["idToSemantics"]
            for idx, label in idToLabels.items():
                for k, v in label.items():
                    if v == robot_name:
                        robot_idx = ast.literal_eval(idx)
                        # print(f"RGBA:{robot_name}:{robot_idx}, {type(robot_idx)}")
            
            if robot_idx is not None:
                b,g,r,a = int(robot_idx[0]), int(robot_idx[1]), int(robot_idx[2]), int(robot_idx[3])
                idx = (a << 24) | (r << 16) | (g << 8) | b
                obj_mask = np.zeros_like(masks, dtype=bool)
                for i in range(len(masks)):
                    if masks[i] == idx:
                        obj_mask[i] = True
                obj_mask = obj_mask.flatten()

                points = points[~obj_mask]
            
        if ignore_grasped_obj:
            grasped_idx = None
            grasped_obj_name = self.grasped_obj_name
            idToLabels = mask_info["idToSemantics"]
            for idx, label in idToLabels.items():
                for k, v in label.items():
                    if v == grasped_obj_name:
                        grasped_idx = ast.literal_eval(idx)

            if grasped_idx is not None:
                b,g,r,a = int(grasped_idx[0]), int(grasped_idx[1]), int(grasped_idx[2]), int(grasped_idx[3])
                idx = (a << 24) | (r << 16) | (g << 8) | b
                obj_mask = np.zeros_like(masks, dtype=bool)           
                for i in range(len(masks)):
                    if masks[i] == idx:
                        obj_mask[i] = True
                obj_mask = obj_mask.flatten()

                points = points[~obj_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        points = np.asarray(pcd_downsampled.points)
        colors = np.asarray(pcd_downsampled.colors).astype(np.uint8)

        return points, colors
    

    def get_cam_extrinsic(self, cam_name):
        return self.cameras[cam_name].get_camera_extrinsic()
    
    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_action = None
        self.latest_action = None

    def apply_action(self, action):
        """
        input a world end effector pose, convert it into joint degrees of freedom and execute the action
        Args:
            action (_type_): np.array([position, quat])

        Returns:
            the world pose of end_effector after the input target pose is executed
        """

        gripper_obj_len = 0.02                # franka gripper抓取物品时的位移，可以根据抓取对象的尺寸动态变化
        action = np.concatenate([action, [gripper_obj_len]])  # 每个action拼接一个gripper位移，以控制gripper状态
        now_action = self.action.move_to_by_dc(action)
        
        now_position, now_quat = now_action[0], now_action[1]

        obs_action = np.concatenate([np.concatenate([now_position, now_quat]), [action[-2]]])
        self.latest_action = obs_action

        self._update_visualizer()
        return now_action

    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        action = np.concatenate([self.latest_action[:-1], [1.0]])
        return self.apply_action(action)

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        action = np.concatenate([self.latest_action[:-1], [0.0]])
        return self.apply_action(action)

    def set_gripper_state(self, gripper_state):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        action = np.concatenate([self.latest_action[:-1], [gripper_state]])
        return self.apply_action(action)
    
    def get_now_action_gripper_pose(self):
        return self.latest_action[:-1]
    
    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:       # 获取抓手状态
            action = np.concatenate([pose, [self.init_action[-1]]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action)
    
    
    def reset_to_default_pose(self):     # ignore
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        pass

    def reset(self):
        """
        Resets the environment and the task. Also updates the visualizer.

        Returns:
            tuple: A tuple containing task descriptions and initial observations.
        """
        assert self.task is not None, "Please load a task first"
        init_franka_pos, init_franka_quat = self.action.compute_end_effector_pose_dy_articulation_pose()
        init_franka_gripper_pose = 1.0
        obs = np.concatenate([np.concatenate([init_franka_pos, init_franka_quat]), [init_franka_gripper_pose]])
        self.init_action = obs
        self.latest_action = obs
        self._update_visualizer()

        return obs

    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return self.init_action[-1]

    def get_ee_pos(self):
        """
        get the world position of end effector by articulation kinematics solver
        Returns:
            np.array(): position
        """
        position, quat = self.action.compute_end_effector_pose_dy_articulation_pose()
        return position

    def get_ee_quat(self):
        """
        get the world orientation of end effector by articulation kinematics solver
        Returns:
            np.array(): quat
        """
        position, quat = self.action.compute_end_effector_pose_dy_articulation_pose()
        return quat 
    
    def get_ee_local_quat(self):
        return self.task.get_observations()["franka"]["gripper_local_orientation"]
    
    def get_gripper_obj_length(self):
        gripper_len = {}
        for obj in self.gripper_obj:
            obj_points, _ = self.get_3d_obs_by_name(obj)
            min = np.min(obj_points, axis=0)
            max = np.max(obj_points, axis=0)
            range = max-min
            print(range)
            size = {"length":range[0], "width":range[1], "higth":range[2]}
            gripper_len[obj] = range[0]
        return gripper_len


def depth2pointscolud(img, extrinsic):
    
    pix_coords = []
    pix_depth = []
    for pix_x in range(640):
        for pix_y in range(480):
            pix_coords.append(img[pix_y][pix_x]*np.array([pix_x, pix_y, 1]).reshape(3,))
            pix_depth.append(img[pix_y][pix_x])

    inv_camera_matrix = np.linalg.inv(np.array([[612.791, 0.0, 321.73],
                                                [0.0, 611.87, 245.06],
                                                [0.0, 0.0, 1.0]]))

    point = []
    for idx in range(len(pix_depth)):
        camera_transform = np.identity(4)
        camera_frame_point = np.matmul(inv_camera_matrix, pix_coords[idx])
        camera_transform[0:3,3] = camera_frame_point

        world_frame = np.matmul(extrinsic, camera_transform)

        point.append(world_frame[0:3,3])

    return point


def get_clip_points(point, min, max, img=None):
    min = np.array(min)
    max = np.array(max)
    point = np.array(point).reshape(-1,3)
    chosen_idx_x = (point[:, 0]>=min[0]) & (point[:, 0]<= max[0])
    chosen_idx_y = (point[:, 1]>=min[1]) & (point[:, 1]<= max[1])
    chosen_idx_z = (point[:, 2]>=min[2]) & (point[:, 2]<= max[2])

    points_cloud = point[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

    if img is not None:
        img = img[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        return img

    return points_cloud

def get_max_min_coord(position, size):
    new = position    # 换原点后新的底面中心店坐标
    min = [new[0]-size[0]/2, new[1]-size[1]/2, new[2]]
    max = [new[0]+size[0]/2, new[1]+size[1]/2, new[2]+size[2]]
    
    return min, max
