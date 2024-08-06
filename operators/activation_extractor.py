from base_classes.base import Operator
from utils.factories import DatasetFactory
from utils.activations import Activations
from utils.utils import create_bounding_boxes_from_predictions, check_detection_matches, draw_bev_bbox
import os 
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from utils.visualizer import Visualizer
from definitions import ROOT_DIR
import pandas as pd

class ActivationExractionOperator(Operator):
    def __init__(self, config):
        self.config = config.extraction
        self.dataset = DatasetFactory().get(**self.config['dataset'])
        self.extract_labels_only = self.config['extract_labels_only']
        self.label_image_config = self.config['method']['label_image']
        
        self.save_dir = self.config['method']['save_dir']
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.activation = Activations(self.config,extract=self.config['active'])
        os.makedirs(self.config['method']['save_dir'], exist_ok=True)
    def execute(self, **kwargs):
        verbose = kwargs.get('verbose', False)
        image_size = (self.label_image_config['width'], self.label_image_config['height'])
        resolution = self.label_image_config['resolution']
        if verbose:
            print("Extracting activations")
            progress_bar = tqdm(total=len(self.dataset))
        for i in range(len(self.dataset)):
            # create a int img with size 
            label_image = np.ones(image_size, dtype=np.uint8) * 0
            data = self.dataset[i]
            cloud, ground_truth_boxes = data['pointcloud'], data['labels']
            file_name = data['file_name']
            if "nus" in self.config['model']['config']: #TODO: might need to change this based on model, as it seems that is the only difference
                cloud.validate_and_update_descriptors(extend_or_reduce = 5)
            result, data = self.activation(cloud.points,file_name)
            predicted_boxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            predicted_scores = result.pred_instances_3d.scores_3d.cpu().numpy()
            score_mask = np.where(predicted_scores >= self.config['score_threshold'])[0] # May require edit later
            filtered_predicted_boxes = predicted_boxes[score_mask]
            prediction_bounding_boxes = create_bounding_boxes_from_predictions(filtered_predicted_boxes)
            matched_bb, unmatched_gt, unmatched_pred = check_detection_matches(ground_truth_boxes, prediction_bounding_boxes)
            for bboxes in matched_bb:
                bbox = bboxes[0]
                bev_bbox = bbox.get_bev_bbox()
                bev_bbox = bev_bbox/resolution
                bev_bbox = bev_bbox.astype(int)
                label_image = draw_bev_bbox(label_image, bev_bbox, 256)
            for bbox in unmatched_gt:
                bev_bbox = bbox.get_bev_bbox()
                bev_bbox = bev_bbox/resolution
                bev_bbox = bev_bbox.astype(int)
                label_image = draw_bev_bbox(label_image, bev_bbox, 128)
            for bbox in unmatched_pred:
                bev_bbox = bbox.get_bev_bbox()
                bev_bbox = bev_bbox/resolution
                bev_bbox = bev_bbox.astype(int)
                label_image = draw_bev_bbox(label_image, bev_bbox, 128)
            # save the image
            label_file = file_name.split('/')[-1].replace('.pcd.bin','.png')
            label_file = os.path.join(self.save_dir,label_file)
            plt.imsave(label_file, label_image, cmap='gray')
            
            if not self.extract_labels_only:
                self.activation.save_single_layer_activation()
            self.activation.clear_activation()
            if verbose:
                progress_bar.update(1)
        # if(self.config['active']):
        
            
                

        
            