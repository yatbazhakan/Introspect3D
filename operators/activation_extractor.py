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
        # try:
        if self.config.get('split',False):
            print("Loading splits")
            self.datasets = {}
            for split in ['train','val','test']:
                self.config['dataset']['split'] = split
                dataset = DatasetFactory().get(**self.config['dataset'])
                self.datasets[split]= dataset
                os.makedirs(os.path.join(self.config['method']['save_dir'],split), exist_ok=True)
                os.makedirs(os.path.join(self.config['method']['save_dir'],split,"features"), exist_ok=True)
        else:
            self.dataset = DatasetFactory().get(**self.config['dataset'])
            os.makedirs(self.config['method']['save_dir'], exist_ok=True)
            os.makedirs(os.path.join(self.config['method']['save_dir'],split,"features"), exist_ok=True)


        self.activation = Activations(self.config,extract=self.config['active'])
        # except:
        #     print("Dataset not found")
        #     exit()
        
    def save_labels(self,split = False):
        if split:
            self.new_dataset.to_csv(os.path.join(ROOT_DIR,
                                             self.config['method']['save_dir'],
                                             split,
                                           self.config['method']['labels_file_name']))
        else:
            self.new_dataset.to_csv(os.path.join(ROOT_DIR,
                                             self.config['method']['save_dir'],
                                           self.config['method']['labels_file_name']))
    def extract(self, verbose = True,split=False):
        self.new_dataset = pd.DataFrame(columns=['name', 'is_missed','missed_objects','total_objects'])
        if verbose:
            print("Extracting activations")
            progress_bar = tqdm(total=len(self.dataset))
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            cloud, ground_truth_boxes, file_name = data['pointcloud'], data['labels'], data['file_name']
            if "nus" in self.config['model']['config']: #TODO: might need to change this based on model, as it seems that is the only difference
                cloud.validate_and_update_descriptors(extend_or_reduce = 5)
            file_name = file_name.replace(self.config['method']['extension'],'')
            result, data = self.activation(cloud.points,file_name)
            # for activation in self.activation.activation_list:
            #     print(activation.shape)
            # exit()
            predicted_boxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            predicted_scores = result.pred_instances_3d.scores_3d.cpu().numpy()
            score_mask = np.where(predicted_scores >= self.config['score_threshold'])[0] # May require edit later
            filtered_predicted_boxes = predicted_boxes[score_mask]
            is_nuscenes = self.config['dataset']['name'] == 'NuScenesDataset'
            is_custom = self.config['dataset']['name'] == 'CustomDataset'
            prediction_bounding_boxes = create_bounding_boxes_from_predictions(filtered_predicted_boxes)
            # vis = Visualizer()
            # vis.visualize(cloud= cloud.points,gt_boxes = ground_truth_boxes,pred_boxes = prediction_bounding_boxes)
            if not is_custom:
        
                matched_boxes, unmatched_ground_truths, unmatched_predictions = check_detection_matches(ground_truth_boxes, prediction_bounding_boxes)
                # print("Matched boxes",len(matched_boxes),"Unmatched boxes",len(unmatched_ground_truths),"Unmatched predictions",len(unmatched_predictions))
                if(len(ground_truth_boxes) > 0):
                    row = {'name':f"{file_name}",'is_missed':len(unmatched_ground_truths) > 0,'missed_objects':len(unmatched_ground_truths),'total_objects':len(ground_truth_boxes)}
                    from pprint import pprint
                    # pprint(row)
                    if split:
                        self.activation.save_multi_layer_activation(split)
                    else:
                        self.activation.save_multi_layer_activation()
                    self.new_dataset = pd.concat([self.new_dataset,pd.DataFrame([row])])
                if verbose:
                    progress_bar.update(1)
            else:
                label = ground_truth_boxes
                row = {'name':f"{file_name}",'is_missed':label,'missed_objects':0,'total_objects':0}
                # print(row)
                if split:
                    self.activation.save_multi_layer_activation(split)
                else:
                    self.activation.save_multi_layer_activation()
                self.new_dataset = pd.concat([self.new_dataset,pd.DataFrame([row])])
                if verbose:
                    progress_bar.update(1)
        # if(self.config['active']):
        if split:
            self.save_labels(split)
        else:
            self.save_labels()
    def execute(self, **kwargs):
        verbose = kwargs.get('verbose', False)
        
        if len(self.datasets) !=0:
            for split,dataset in self.datasets.items():
                self.dataset = dataset
                self.extract(verbose,split)
        else:
            self.extract(verbose)

                

        
            