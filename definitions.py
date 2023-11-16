import os 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
CONFIG_DIR = os.path.join(ROOT_DIR,'configs')
DATASETS_DIR = os.path.join(ROOT_DIR,'datasets')
OPERATORS_DIR = os.path.join(ROOT_DIR,'operators')
UTILS_DIR = os.path.join(ROOT_DIR,'utils')

KITTI_CLASSES = {  "Car" : 0,
    "Van" : 0 ,
    "Truck": 0 ,
    "Pedestrian" : 1,
    "Tram": 0  }

CITYSCAPES_COLORS={
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "pole": (153, 153, 153),
    "traffic_light": (250, 170, 30),
    "traffic_sign": (220, 220, 0),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "person": (220, 20, 60),
    "rider": (255, 0, 0),
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "bus": (0, 60, 100),
    "train": (0, 80, 100),
    "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32)
}

CITYSCAPES_INDEX_TO_COLOR= {i:color for i,(name,color) in enumerate(CITYSCAPES_COLORS.items())}