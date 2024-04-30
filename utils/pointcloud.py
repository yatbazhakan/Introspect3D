import numpy as np
class PointCloud:
    def __init__(self, points,model_req_dim = 4) -> None:
        self.raw_points = points
        # self.validate_and_update_descriptors(extend_or_reduce=model_req_dim)
        
    def validate_and_update_descriptors(self, **kwargs):
        if self.raw_points.shape[1] < 3:
            raise ValueError("Points array must have at least 3 columns for X, Y, and Z coordinates.")
        extend_or_reduce = kwargs.get('extend_or_reduce', 4)
        if self.raw_points.shape[1] == extend_or_reduce:
            self.points = self.raw_points
        elif self.raw_points.shape[1] < extend_or_reduce and self.points.shape[1] != extend_or_reduce:
            self.points = np.hstack((self.raw_points, np.zeros((len(self.raw_points), extend_or_reduce - self.raw_points.shape[1]))))
        elif self.raw_points.shape[1] > extend_or_reduce:
            self.points = self.raw_points[:, :extend_or_reduce]
        # Further logic to handle cases where M=4 or 5 can be added here
    
    def __len__(self):
        return len(self.points)
    def set_points_as_raw(self):
        self.points = self.raw_points
    def get_points(self):
        return self.points
    
    def get_raw_points(self):
        return self.raw_points
    
    def convert_to_kitti_points(self):
        self.validate_and_update_descriptors()
        
    def convert_to_nuscenes_points(self):
        if self.raw_points.shape[1] == 3:
            self.points = np.hstack((self.raw_points, np.ones((len(self.raw_points), 1))))
        elif self.raw_points.shape[1] == 4:
            self.points = self.raw_points  # Assuming the 4th column is the time/intensity
    
    def transform_to_coordinate_frame(self, transform):
        if self.points.shape[1] == 3:
            self.points = np.dot(transform, np.hstack((self.points, np.ones((len(self.points), 1)))).T).T[:, :3]
        else:
            self.points = np.dot(transform, self.points.T).T
    
    def filter_pointcloud(self, filter_type, **kwargs):
        filter_strategy = filter_type.value(**kwargs)
        self.points = filter_strategy.filter_pointcloud(self.points)
# class PointCloud:
#     def __init__(self,points) -> None:
#         self.raw_points = points
#         self.points = points[:,:3]
        
#     def __len__(self):
#         return len(self.points)
    
#     def get_points(self):
#         return self.points
    
#     def get_raw_points(self):
#         return self.raw_points
    
#     def convert_to_kitti_points(self):
#         #Return first three columns of points as xyz
#         self.points=  self.raw_points[:,:3]
        
#     def convert_to_nuscenes_points(self):
#         #Add an extra column of ones to points as itrequires time on top of xyzi
#         self.points = np.hstack((self.raw_points,np.ones((len(self.raw_points),1))))
        
#     def transform_to_coordinate_frame(self,transform):
#         #Transform points to new coordinate frame
#         self.points = np.dot(transform,self.points.T).T
        
#     def filter_pointcloud(self,filter,**kwargs): #0 for rectangle, 1 for ellipse
#         filter_strategy = filter.value(**kwargs)
#         self.points = filter_strategy.filter_pointcloud(self.points)
        
    