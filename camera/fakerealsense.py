import omni
import numpy as np

from pxr import Sdf, Gf
import omni.replicator.core as rep
from omni.isaac.sensor import Camera
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims.xform_prim import XFormPrim


class FakeRealSense:
    def __init__(self, 
                 name = "cam",
                 position:np.array=np.zeros(3), 
                 orientation = np.array([0.3, -0.95372, 0., 0.]),
                 resolution:tuple=(640, 480),
                 frequency:int=30,
                 camera_matrix:list=[[612.791, 0.0, 321.73], [0.0, 611.87, 245.06], [0.0, 0.0, 1.0]],      # 内参
                 focal_length:float=1.93,
                 clip_range:tuple=(0.03, 1000.),
                 base_link:str="/World")->None:
        """Create a fake camera in isaac sim

        Args:
            position (np.array, optional): position of the camera. Defaults to np.zeros(3).
            quat (np.array, optional): orientation of the robot, w last. Defaults to np.asarray([0., 0., 0., 1]).
            resolution (tuple, optional): size of the image. Defaults to (640, 480).
            frequency (int, optional): fps. Defaults to 30.
            camera_matrix (list, optional): camera intrinsic matrix. Defaults to [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]].
            focal_length (float, optional): camera focal length in mm. Defaults to 1.93.
            clip_range (tuple, optional): clipping range in m. Defaults to (0.01, 10.).
        """
        self.name = name
        self.resolution = resolution

        create_prim(f"{base_link}/camera_axis")
        self.camera_frame = XFormPrim(
            prim_path=f"{base_link}/camera_axis",)
        
        self.camera_frame.set_local_pose(translation=position,
                                         orientation=orientation)
        
        create_prim(f"{base_link}/camera_axis/camera_direction")
        camera_direction = XFormPrim(
            prim_path=f"{base_link}/camera_axis/camera_direction")
        
        camera_direction.set_local_pose(orientation=[0.5, 0.5, -0.5, 0.5])

        self.camera = Camera(
            prim_path=f"{base_link}/camera_axis/camera_direction/camera",
        #    position=np.array([0., 0., 0.]),
            frequency=20,
            resolution=resolution,
        #    orientation=[0.5, 0.5, -0.5, -0.5]#rot_utils.euler_angles_to_quats(np.array([-71.045, 10.648, 176.369]), degrees=True),
        )
        self.camera.initialize()
            
        width, height = resolution

        ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix

        aspect_ratio = height / width
        horizontal_aperture = width * focal_length / fx
        vertical_aperture = aspect_ratio * horizontal_aperture

        horizontal_aperture_offset = (cx - width / 2.) / fx
        vertical_aperture_offset = (cy - height / 2) / fy

        # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
        self.camera.set_focal_length(focal_length / 10.0)
        # self.camera.set_focus_distance(10.)
        # self.camera.set_lens_aperture(f_stop * 100.0)
        self.camera.set_horizontal_aperture(horizontal_aperture / 10.0)
        self.camera.set_vertical_aperture(vertical_aperture / 10.0)
        # self.camera.set_horizontal_aperture_offset(horizontal_aperture_offset / 10.0)
        # self.camera.set_vertical_aperture_offset(vertical_aperture_offset / 10.0)

        self.camera.set_clipping_range(clip_range[0], clip_range[1])

        # add depth imformation
        self.camera.add_distance_to_image_plane_to_frame()
        # make sure camera can get the semantic informatiom
        self.camera.add_instance_id_segmentation_to_frame()
        # you can read colorize mask use this customize function as will as binary mask by comment function
        self.add_colorize_instance_segmentation_to_frame()
        # self.camera.add_instance_id_segmentation_to_frame()
        self.camera.add_semantic_segmentation_to_frame()
        # add 3D bounding box
        self.camera.add_bounding_box_3d_to_frame()

    def add_colorize_instance_segmentation_to_frame(self):
        """Attach the instance_segmentation annotator to this camera.
        The main difference between instance id segmentation and instance segmentation are that instance segmentation annotator goes down the hierarchy to the lowest level prim which has semantic labels, which instance id segmentation always goes down to the leaf prim.
        The instance_segmentation annotator returns:

            np.array
            shape: (width, height, 1) or (width, height, 4) if `colorize` is set to true
            dtype: np.uint32 or np.uint8 if `colorize` is set to true

        See more details: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#instance-segmentation
        """
        if self.camera._custom_annotators["instance_segmentation"] is None:
            self.camera._custom_annotators["instance_segmentation"] = rep.AnnotatorRegistry.get_annotator(
                "instance_segmentation",{"colorize": True}
            )
            self.camera._custom_annotators["instance_segmentation"].attach([self.camera._render_product_path])
        self.camera._current_frame["instance_segmentation"] = None
        return
     
    def get_instance_id_segmentation(self):
        """Attach the instance_id_segmentation annotator to this camera.
        The instance_id_segmentation annotator returns:

            np.array
            shape: (width, height, 1) or (width, height, 4) if `colorize` is set to true
            dtype: np.uint32 or np.uint8 if `colorize` is set to true

        See more details: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#instance-id-segmentation
        """
        mask = self.camera._custom_annotators["instance_id_segmentation"].get_data()
        return mask
    
    def get_instance_segmentation(self):
        """
        Returns:
        {"data":mask img data, "info":sementic information}
        """
        mask = self.camera._custom_annotators["instance_segmentation"].get_data()
        return mask
    
    def get_semantic_segmentation(self):
        mask = self.camera._custom_annotators["semantic_segmentation"].get_data()
        return mask
    
    def get_bbx_3d(self):
        bbx_3d = self.camera._custom_annotators["bounding_box_3d"].get_data()
        return bbx_3d
    
    def get_camera_frame_point(self):
        return self.camera.get_pointcloud()
    
    def get_cam(self):
        return self.camera
    
    def read(self)->tuple:
        """get rgb image and depth image

        Returns:
            tuple: rgb image of shape (width X height X 3), depth image of shape (width X height)
        """
        self.camera.get_current_frame()       # 得到当前帧数据

        color_image = self.camera.get_rgba()[:, :, :3]
        depth_image = self.camera.get_depth()

        return color_image, depth_image
    
    def get_camera_extrinsic(self):
        extrinsic_matrix = np.identity(4)
        # print(f"{self.camera_frame}")
        # input()
        position, orientation = self.camera_frame.get_world_pose()
        extrinsic_matrix[0:3,3] = position

        quat = np.zeros(4)
        quat[-1] = orientation[0]
        quat[0:3] = orientation[1:]

        extrinsic_matrix[0:3,0:3] = R.from_quat(quat).as_matrix()

        return extrinsic_matrix
    
    def get_world_points(self):
        """
        Use 2d points (u, v) corresponds to the pixel coordinates and depth corresponds to each of the pixel coords to get world points. 
        2d points shape is (n, 2) where n is the number of points. depth shape is (n,)
        
        Returns:
            np.ndarray: (n, 3) 3d points (X, Y, Z) in world frame. shape is (n, 3) where n is the number of points.
        """
        depth = self.camera.get_depth()
        depth = np.transpose(depth)                    # depth, RGB, mask must has same shape
        pix_coords = []
        for pix_x in range(self.resolution[0]):        # get 2d pixel points
            for pix_y in range(self.resolution[1]):
                pix_coords.append(np.array([pix_x, pix_y]))   
        pix_coords = np.array(pix_coords).reshape(-1, 2)    
        points_world = self.camera.get_world_points_from_image_coords(points_2d=pix_coords, depth=depth.flatten())

        return points_world
    
    def get_points_by_depth(self):
        """
        methods to convert depth image into point cloud
        Returns:
            np.ndarray: (n, 3) 3d points (X, Y, Z) in world frame. shape is (n, 3) where n is the number of points.
        """
        depth = self.camera.get_depth()
        # print(depth.shape)    # (798, 1024)

        extrinsic = np.identity(4)
        position, orientation = self.camera_frame.get_world_pose()
        extrinsic[0:3,3] = position

        quat = np.zeros(4)
        quat[-1] = orientation[0]
        quat[0:3] = orientation[1:]

        extrinsic[0:3,0:3] = R.from_quat(quat).as_matrix()

        pix_coords = []
        pix_depth = []
        for pix_x in range(self.resolution[0]):
            for pix_y in range(self.resolution[1]):
                pix_coords.append(depth[pix_y][pix_x]*np.array([pix_x, pix_y, 1]).reshape(3,))
                pix_depth.append(depth[pix_y][pix_x])

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
        point = np.array(point).reshape(-1, 3)

        return point
