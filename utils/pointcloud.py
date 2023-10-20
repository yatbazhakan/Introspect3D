import numpy as np
from filter import *

class PointCloud:
    def __init__(self,points) -> None:
        self.raw_points = points
        self.points = points
        
    def __len__(self):
        return len(self.points)
    
    def get_points(self):
        return self.points
    
    def get_raw_points(self):
        return self.raw_points
    
    def convert_to_kitti_points(self):
        #Return first three columns of points as xyz
        self.points=  self.raw_points[:,:3]
        
    def convert_to_nuscenes_points(self):
        #Add an extra column of ones to points as itrequires time on top of xyzi
        self.points = np.hstack((self.raw_points,np.ones((len(self.raw_points),1))))
        
    def transform_to_coordinate_frame(self,transform):
        #Transform points to new coordinate frame
        self.points = np.dot(transform,self.points.T).T
        
    def filter_pointcloud(self,filter: FilterType,**kwargs): #0 for rectangle, 1 for ellipse
        filter_strategy = filter.value(**kwargs)
        self.points = filter_strategy.filter_pointcloud(self.points)
        
    