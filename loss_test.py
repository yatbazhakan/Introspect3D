# import torch
# import torch.nn as nn
# from modules.loss import FocalLoss
# import numpy as np
# import os 
# import pandas as pd
# from glob import glob
# from definitions import ROOT_DIR
# from utils.metrics import TPRatFPR, FPRatTPR

# # Simple deterministi

# y_true_multiclass = torch.tensor([0, 0, 1, 1, 1, 0, 0, 1])
# y_scores_multiclass = torch.tensor([
#     [0.9, 0.1],  # High probability for class 0
#     [0.8, 0.2],  # High probability for class 0
#     [0.2, 0.8],  # High probability for class 1
#     [0.6, 0.4],  # Higher probability for class 0
#     [0.1, 0.9],  # High probability for class 1
#     [0.7, 0.3],  # Higher probability for class 0
#     [0.4, 0.6],  # Higher probability for class 1
#     [0.3, 0.7]   # Higher probability for class 1
# ])

# tpratfpr = TPRatFPR()
# tpratfpr.update(y_true_multiclass, y_scores_multiclass)
# print(tpratfpr.compute())
# fprattpr = FPRatTPR()
# fprattpr.update(y_true_multiclass, y_scores_multiclass)
# print(fprattpr.compute())


# import torch
# from sklearn.metrics import roc_curve

# # Provided data
# y_true_multiclass = torch.tensor([0, 0, 1, 1, 1, 0, 0, 1])
# y_scores_multiclass = torch.tensor([
#     [0.9, 0.1],
#     [0.8, 0.2],
#     [0.2, 0.8],
#     [0.6, 0.4],
#     [0.1, 0.9],
#     [0.7, 0.3],
#     [0.4, 0.6],
#     [0.3, 0.7]
# ])

# # We will focus on the scores for class 1 (the second column)
# y_scores_class1 = y_scores_multiclass[:, 1]

# # Calculate ROC curve
# fpr, tpr, thresholds = roc_curve(y_true_multiclass, y_scores_class1)

# # Find the closest point on the ROC curve with TPR>=0.95
# closest_tpr_index = np.where(tpr >= 0.95)[0][0]
# fpr_at_95tpr = fpr[closest_tpr_index]

# # Find the closest point on the ROC curve with FPR<=0.05
# closest_fpr_index = np.where(fpr <= 0.05)[0][-1]  # Get the last index where FPR is below 0.05
# tpr_at_05fpr = tpr[closest_fpr_index]

# print(fpr_at_95tpr, tpr_at_05fpr)
# # For binary classification, there are 2 classes
# # num_classes = 2

# # # Define specific predictions and true labels for a batch size of 3
# # predictions = torch.tensor([
# #     [2.0, 1.0],  # Likely classified as class 0
# #     [1.0, 3.0],  # Likely classified as class 1
# #     [0.5, 0.5]   # Equally likely for both classes
# # ])

# # true_labels = torch.tensor([0, 1, 0]) # Corresponding true class labels

# # # Using CrossEntropyLoss for binary classification
# # criterion = nn.CrossEntropyLoss()
# # criterion2 = FocalLoss(gamma=0,alpha=torch.tensor([1.0,1.0]))
# # criterion4 = torch.hub.load('adeelh/pytorch-multi-class-focal-loss',
# # 	model='FocalLoss',
# # 	alpha=torch.tensor([1.0,1.0]),
# # 	gamma=0,
# # 	reduction='mean',
# # 	force_reload=False)
# # # Calculate the loss
# # loss = criterion(predictions, true_labels)
# # loss2 = criterion2(predictions, true_labels)
# # loss4 = criterion4(predictions, true_labels)

# # print("Loss:", loss.item())
# # # Print the loss
# # # print("FocalLossCustom Loss:", loss2.item())
# # print("FocalLoss Loss:", loss2.item())
# # print("FocalLoss2 Loss:", loss4.item())
from utils.converter import DatasetConverter
import os
kitti_root_path = r"/mnt/ssd2/kitti/"
os.makedirs(os.path.join(kitti_root_path,'coco'),exist_ok=True)
converter = DatasetConverter(root_dir=kitti_root_path,output_dir=os.path.join(kitti_root_path,'coco'),dataset_type="KITTI")
converter.split_dataset()
# converter.add_category('car', 1)
# converter.add_category('pedestrian', 2)
# converter.add_category('truck', 3)
# converter.add_category('bus', 4)
# converter.add_category('train', 5)
# converter.convert('BDD')
# converter.add_category('Car', 1)
# converter.add_category('Pedestrian', 2)
# converter.add_category('Cyclist', 3)
# converter.convert('KITTI')