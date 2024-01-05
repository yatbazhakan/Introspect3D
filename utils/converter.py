import json
import os
from PIL import Image

class DatasetConverter:
    def __init__(self, **kwargs):
        self.configs = kwargs['image_dir']
        self.coco = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.ann_id = 1

    def add_category(self, name, id_):
        self.coco['categories'].append({
            'id': id_,
            'name': name,
            'supercategory': 'object'
        })

    def convert(self, dataset_type):
        self.output_dir = self.configs['output_dir']
        if dataset_type == 'KITTI':
            self.convert_kitti()
        elif dataset_type == 'BDD':
            self.convert_bdd()
        else:
            raise ValueError("Unsupported dataset type")

        # Write the COCO data to a JSON file
        with open(os.path.join(self.output_dir, f'{dataset_type}_coco_format.json'), 'w') as json_file:
            json.dump(self.coco, json_file)

        print(f"Conversion completed for {dataset_type}!")

    def convert_kitti(self):
        self.image_dir = os.path.join(self.configs['root_dir'], 'training', 'image_2')
        self.label_dir = os.path.join(self.configs['root_dir'], 'training', 'label_2')
        for filename in os.listdir(self.label_dir):
            # Extracting image information
            image_id = filename.split('.')[0]
            image_path = os.path.join(self.image_dir, image_id + '.png')
            image = Image.open(image_path)
            image_width, image_height = image.size

            # Image entry
            self.coco['images'].append({
                "file_name": image_id + '.png',
                "height": image_height,
                "width": image_width,
                "id": image_id
            })

            # Annotations
            with open(os.path.join(self.label_dir, filename), 'r') as file:
                for line in file:
                    line = line.strip().split(' ')
                    category_name = line[0]
                    if category_name in [category['name'] for category in self.coco['categories']]:
                        # KITTI bounding box format: left, top, right, bottom
                        left, top, right, bottom = map(float, line[4:8])
                        width = right - left
                        height = bottom - top
                        
                        # COCO bounding box format: [min_x, min_y, width, height]
                        self.coco['annotations'].append({
                            "id": self.ann_id,
                            "image_id": image_id,
                            "category_id": next(cat['id'] for cat in self.coco['categories'] if cat['name'] == category_name),
                            "bbox": [left, top, width, height],
                            "area": width * height,
                            "iscrowd": 0
                        })
                        self.ann_id += 1
                        
    def convert_bdd(self):
        # Load BDD annotations
        self.image_dir = os.path.join(self.configs['root_dir'], 'images', '100k', 'train')
        self.label_dir = os.path.join(self.configs['root_dir'], 'labels')
        with open(os.path.join(self.label_dir, 'bdd100k_labels_images_train.json'), 'r') as f:
            bdd_annotations = json.load(f)

        for ann in bdd_annotations:
            image_id = ann['name'].split('.')[0]
            image_path = os.path.join(self.image_dir, ann['name'])
            image = Image.open(image_path)
            image_width, image_height = image.size

            # Add image info to COCO dataset
            self.coco['images'].append({
                "file_name": ann['name'],
                "height": image_height,
                "width": image_width,
                "id": image_id
            })

            # Convert BDD annotations to COCO format
            for label in ann['labels']:
                if label['category'] in [category['name'] for category in self.coco['categories']]:
                    # BDD bounding box format: [x1, y1, x2, y2]
                    bbox = label['box2d']
                    if not bbox:
                        continue
                    left, top, right, bottom = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    width = right - left
                    height = bottom - top

                    # COCO bounding box format: [min_x, min_y, width, height]
                    self.coco['annotations'].append({
                        "id": self.ann_id,
                        "image_id": image_id,
                        "category_id": next(cat['id'] for cat in self.coco['categories'] if cat['name'] == label['category']),
                        "bbox": [left, top, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    self.ann_id += 1