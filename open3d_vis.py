import open3d as o3d
import os
from glob import glob
import numpy as np
def make_point_cloud(path):
    cloud = np.load(path)
    points = cloud[:, :3]            
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(points), 3)))
    return pcd
def load_point_clouds(folder_path):
    """Load all point cloud files from the seeeeepecified folder."""
    files = glob(os.path.join(folder_path, 'lidar','*.npy'))

    point_clouds = [make_point_cloud(file) for file in files]
    return point_clouds

def visualize_point_clouds(point_clouds):
    """Visualize point clouds and switch between them using the spacebar."""
    if not point_clouds:  # Check if list is empty
        print("No point clouds found in the directory.")
        return

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    current_index = [0]  # Using a list to hold the index as a mutable object

    def load_next_point_cloud(vis):
        """Callback function to load the next point cloud."""
        current_index[0] = (current_index[0] + 1) % len(point_clouds)
        vis.clear_geometries()
        vis.add_geometry(point_clouds[current_index[0]], reset_bounding_box=True)

    vis.register_key_callback(ord(' '), load_next_point_cloud)  # Bind spacebar to switch point clouds
    vis.add_geometry(point_clouds[current_index[0]], reset_bounding_box=True)

    vis.run()  # Run the visualizer
    vis.destroy_window()  # Clean up after closing the window

# Usage
if __name__ == '__main__':
    folder_path = '/media/yatbaz_h/Jet/HYY/'
    point_clouds = load_point_clouds(folder_path)
    visualize_point_clouds(point_clouds)
