import open3d as o3d
import numpy as np
import cv2 as cv

from redrawing.components.stage import Stage
from redrawing.data_interfaces.bodypose import BodyPose
from redrawing.data_interfaces.depth_map import Depth_Map
from redrawing.data_interfaces.image import Image



class PCR_Viewer(Stage):
    
    configs_default = {"bodypose":False}

    def __init__(self, configs={}):
        super().__init__(configs)
        self.addInput("depth", Depth_Map)
        self.addInput("rgb", Image)

        if self._configs["bodypose"]:
            self.addInput("bodypose", BodyPose)
            self.addInput("bodypose_list",list)
        
        self.camera_intrinsics = np.eye(3,dtype=np.float64)
        self.calib_size = np.array([300,300])
        


    def setup(self):
        self._config_lock = True

        self.depth = None
        self.rgb = None
        self.started = False

        self.point_cloud = None

        if self._configs["bodypose"]:
            self.bodypose_geometry = []

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()


        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.vis.add_geometry(origin)


        


        

    def process(self):
        
        changed = False

        bodyposes = []

        if self.has_input("depth"):
            self.depth = self._getInput("depth").depth
            changed = True
        if self.has_input("rgb"):
            self.rgb = self._getInput("rgb").image
            changed = True

        if self._configs["bodypose"] and self.has_input("bodypose"):
            bodyposes.append(self._getInput("bodypose"))

        if self._configs["bodypose"] and self.has_input("bodypose_list"):
            for bd in self._getInput("bodypose_list"):
                bodyposes.append(bd)
        


        if self.depth is not None and self.rgb is not None and changed:
            img = cv.cvtColor(self.rgb, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (self.depth.shape[1], self.depth.shape[0]))

            depth = (1000.0*self.depth).astype(np.uint16)

            scale = np.array([1,1],dtype=np.float64)
            scale[0] = self.depth.shape[0] / self.calib_size[0]
            scale[1] = self.depth.shape[1] / self.calib_size[1]

            K = self.camera_intrinsics.copy()
            K[0] *= scale[0]
            K[1] *= scale[1]

            rgb_o3d = o3d.geometry.Image(img)
            depth_o3d = o3d.geometry.Image(depth)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)

            pinhole_model = o3d.camera.PinholeCameraIntrinsic(img.shape[0],img.shape[1],
                                                            K[0][0],
                                                            K[1][1],
                                                            K[0][2],
                                                            K[1][2])



            if not self.started:
                self.point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_model)

                self.vis.add_geometry(self.point_cloud)
                self.started = True
            else:
                new_pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_model)
                self.point_cloud.points = new_pc.points
                self.point_cloud.colors = new_pc.colors
                
                
                self.vis.update_geometry(self.point_cloud)
        
        if self.depth is not None and changed:
            depht_img = 256*self.depth/np.max(self.depth)
            cv.imshow("depth",depht_img)

        if bodyposes != []:
            for geometry in self.bodypose_geometry:
                self.vis.remove_geometry(geometry, False)
                pass

            for bd in bodyposes:
                for name in BodyPose.keypoints_names:
                    kp = bd.get_keypoint(name)

                    if np.isinf(kp[0]):
                        continue
                        
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)

                    #print(kp*1000)

                    sphere.translate([kp[0],kp[1],kp[2]])
                    sphere.paint_uniform_color([1,0,1])

                    self.vis.add_geometry(sphere, False)

                    self.bodypose_geometry.append(sphere)


        self.vis.poll_events()
        self.vis.update_renderer()

        cv.waitKey(1)