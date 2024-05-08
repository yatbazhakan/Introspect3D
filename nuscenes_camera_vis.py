import sys
import glob
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, Qt
import numpy as np
import argparse
import pickle
import cv2
from nuscenes.nuscenes import NuScenes

# Usage

ROOT_DIR = '/mnt/ssd2/nuscenes'
class Viewer(QWidget):
    def __init__(self, nuscenes_data, tokens):
        super().__init__()
        self.path = "/media/yatbaz_h/Jet/HYY"
        self.setWindowTitle('PointCloud with Images')
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()
        self.idx = 0
        self.nuscenes_data = nuscenes_data
        self.tokens = tokens
        self.iterative_token = None
        self.set_scene(tokens[0])
        
    def set_scene(self,token):
        self.scene = nuscenes_data.get('scene', token)
        self.iterative_token = self.scene['first_sample_token']
    def get_frame(self):
        token = self.iterative_token
        sample_record = nuscenes_data.get('sample', token)
        camera_front_token = sample_record['data']['CAM_FRONT']
        camera_back_token = sample_record['data']['CAM_BACK']
        camera_front_left_token = sample_record['data']['CAM_FRONT_LEFT']
        camera_front_right_token = sample_record['data']['CAM_FRONT_RIGHT']
        camera_back_left_token = sample_record['data']['CAM_BACK_LEFT']
        camera_back_right_token = sample_record['data']['CAM_BACK_RIGHT']
        
        front = nuscenes_data.get('sample_data', camera_front_token)
        back = nuscenes_data.get('sample_data', camera_back_token)
        front_left = nuscenes_data.get('sample_data', camera_front_left_token)
        front_right = nuscenes_data.get('sample_data', camera_front_right_token)
        back_left = nuscenes_data.get('sample_data', camera_back_left_token)
        back_right = nuscenes_data.get('sample_data', camera_back_right_token)

        self.iterative_token = sample_record['next']
        return front, back, front_left, back_left,front_right , back_right
    def initUI(self):
        self.main_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.middle_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.left_labels = [QLabel() for _ in range(2)]
        for label in self.left_labels:
            self.left_layout.addWidget(label)
        self.middle_labels = [QLabel() for _ in range(2)]
        for label in self.middle_labels:
            self.middle_layout.addWidget(label)
        self.right_labels = [QLabel() for _ in range(2)]
        for label in self.right_labels:
            self.right_layout.addWidget(label)

        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.middle_layout)
        self.main_layout.addLayout(self.right_layout)
        self.setLayout(self.main_layout)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.update_lidar_and_images()

    def update_lidar_and_images(self):
        

        cameras  = self.get_frame()
        labels = self.middle_labels +self.left_labels +  self.right_labels

        for camera, label in zip(cameras, labels):
            
            camera_data = camera['filename']
            camera_data= os.path.join(ROOT_DIR, camera_data)
            # cv2_image= cv2.imread(camera_data)
            # print(camera_data)
            pixmap = QPixmap(camera_data)
            label.setPixmap(pixmap.scaled(1224 // 3, 1024 // 3))


def arg_parse():
    parser = argparse.ArgumentParser(description='Visualize point clouds')
    parser.add_argument('-n', "--names",nargs='+', default=[], help='Names of the scenes to visualize')
    return parser.parse_args()
if __name__ == '__main__':
    root_dir = '/mnt/ssd2/nuscenes'
    version = 'v1.0-trainval'
    args = arg_parse()

    app = QApplication(sys.argv)
    nuscenes_data= NuScenes(version=version, dataroot=root_dir, verbose=True)

    viewer = Viewer(nuscenes_data, args.names)
    viewer.show()
    sys.exit(app.exec_())

# import sys
# import glob
# import os
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
# from PyQt5.QtGui import QPixmap
# from PyQt5.QtCore import QTimer
# from PyQt5.QtCore import Qt
# import open3d as o3d
# import numpy as np
# class PointCloudViewer:
#     def __init__(self):
#         self.vis = o3d.visualization.Visualizer()
#         self.vis.create_window(visible=False)  # Set visible to True if you want to see the window
#         self.pcd = o3d.geometry.PointCloud()
#         self.initialized = False
#         self.path = "/media/yatbaz_h/Jet/HYY"

#     def update_point_cloud(self, npy_file):
#         lidar = np.load(npy_file)
#         self.pcd.points = o3d.utility.Vector3dVector(lidar[:, :3])
#         self.pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(lidar), 3)))
#         if not self.initialized:
#             print("Initialized")
#             self.vis.create_window("Open3D", 640, 480,visible=True)
#             self.initialized = True
#             self.vis.add_geometry(self.pcd)
#         self.vis.update_geometry(self.pcd)
#         self.vis.poll_events()
#         self.vis.update_renderer()


#     def close(self):
#         self.vis.destroy_window()
# class Viewer(QWidget):
#     def __init__(self, point_cloud_viewer):
#         super().__init__()
#         # self.point_cloud_viewer = point_cloud_viewer
#         self.vis = o3d.visualization.Visualizer()
#         self.vis.create_window(visible=False)  # Set visible to True if you want to see the window
#         self.pcd = o3d.geometry.PointCloud()
#         self.initialized = False
#         self.path = "/media/yatbaz_h/Jet/HYY"
#         self.setWindowTitle('PointCloud with Images')
#         self.setGeometry(100, 100, 1200, 800)
#         self.initUI()
#         self.idx = 0

#     def initUI(self):
#         self.main_layout = QHBoxLayout()
#         self.left_layout = QVBoxLayout()
#         self.middle_layout = QVBoxLayout()
#         self.right_layout = QVBoxLayout()
        
#         # Setup left, middle, and right image layouts
#         self.left_labels = [QLabel() for _ in range(2)]
#         for label in self.left_labels:
#             self.left_layout.addWidget(label)
#         self.middle_labels = [QLabel() for _ in range(2)]
#         for label in self.middle_labels:
#             self.middle_layout.addWidget(label)
#         self.right_labels = [QLabel() for _ in range(2)]
#         for label in self.right_labels:
#             self.right_layout.addWidget(label)

#         # Adding layouts to main layout
#         self.main_layout.addLayout(self.left_layout)
#         self.main_layout.addLayout(self.middle_layout)
#         self.main_layout.addLayout(self.right_layout)
#         self.setLayout(self.main_layout)

#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key_Space:  # Spacebar to trigger update
#             self.update_lidar_and_images()

#     def update_lidar_and_images(self):
#         # Assuming *.npy file has LiDAR data
#         lidar_folder = os.path.join(self.path, 'lidar')
#         lidar_files = glob.glob(os.path.join(lidar_folder, '*.npy'))

#         # Usssssssspdate images
#         folders = ['camera05', 'camera06', 'camera03', 'camera07', 'camera02', 'camera01', 'camera04']
#         labels = self.left_labels + self.middle_labels + self.right_labels
#         for folder, label in zip(folders, labels):
#             image_folder = os.path.join(self.path, folder)
#             image_files = glob.glob(os.path.join(image_folder, '*.png'))
#             if image_files and self.idx < len(image_files):
#                 latest_image = image_files[self.idx]
#                 pixmap = QPixmap(latest_image)
#                 label.setPixmap(pixmap.scaled(1224 // 3, 1024 // 3))
#         lidar = np.load(lidar_files[self.idx])
#         self.pcd.points = o3d.utility.Vector3dVector(lidar[:, :3])
#         self.pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(lidar), 3)))
#         if not self.initialized:
#             print("Initialized")
#             self.vis.create_window("Open3D", 640, 480,visible=True)
#             self.initialized = True
#             self.vis.add_geometry(self.pcd)
#         self.vis.update_geometry(self.pcd)
#         self.vis.poll_events()
#         self.vis.update_renderer()
#         self.idx = (self.idx + 1) % len(image_files)


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     pc_viewer = PointCloudViewer()
#     viewer = Viewer(pc_viewer)
#     viewer.show()
#     sys.exit(app.exec_())

