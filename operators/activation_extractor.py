from base_classes.base import Operator
from utils.factories import DatasetFactory
from utils.activations import Activations
from utils.utils import create_bounding_boxes_from_predictions, check_detection_matches
import os 
import numpy as np
from tqdm.auto import tqdm
from utils.visualizer import Visualizer
from definitions import ROOT_DIR
import pandas as pd
class ActivationExractionOperator(Operator):
    def __init__(self, config):
        self.config = config.extraction
        self.dataset = DatasetFactory().get(**self.config['dataset'])
        self.activation = Activations(self.config,extract=self.config['active'])
        os.makedirs(self.config['method']['save_dir'], exist_ok=True)
    def save_labels(self):
        self.new_dataset.to_csv(os.path.join(ROOT_DIR,
                                             self.config['method']['save_dir'],
                                             self.config['method']['labels_file_name']))
    def execute(self, **kwargs):
        verbose = kwargs.get('verbose', False)
        self.new_dataset = pd.DataFrame(columns=['name', 'is_missed','missed_objects','total_objects'])

        if verbose:
            print("Extracting activations")
            progress_bar = tqdm(total=len(self.dataset))
        for i in range(len(self.dataset)):
            # vis = Visualizer()
            cloud, ground_truth_boxes, file_name = self.dataset[i]
            file_name = file_name.replace('.png','')
            result, data = self.activation(cloud.points,file_name)
            predicted_boxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            predicted_scores = result.pred_instances_3d.scores_3d.cpu().numpy()
            score_mask = np.where(predicted_scores >= self.config['score_threshold'])[0] # May require edit later
            filtered_predicted_boxes = predicted_boxes[score_mask]
            prediction_bounding_boxes = create_bounding_boxes_from_predictions(filtered_predicted_boxes)
            # vis.visualize(cloud= cloud.points,gt_boxes = ground_truth_boxes,pred_boxes = prediction_bounding_boxes)
            matched_boxes, unmatched_ground_truths, unmatched_predictions = check_detection_matches(ground_truth_boxes, prediction_bounding_boxes)
            # print("Matched boxes",len(matched_boxes),"Unmatched boxes",len(unmatched_ground_truths),"Unmatched predictions",len(unmatched_predictions))
            if(len(ground_truth_boxes) > 0):
                row = {'name':f"{file_name}",'is_missed':len(unmatched_ground_truths) > 0,'missed_objects':len(unmatched_ground_truths),'total_objects':len(ground_truth_boxes)}
                self.new_dataset = pd.concat([self.new_dataset,pd.DataFrame([row])])
            if verbose:
                progress_bar.update(1)
            
            self.save_labels() #might be dependent to some condition in future
            