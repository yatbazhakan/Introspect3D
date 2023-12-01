import torch
import torch.nn as nn
from modules.loss import FocalLoss
import numpy as np
import os 
import pandas as pd
from glob import glob
from definitions import ROOT_DIR


# For binary classification, there are 2 classes
# num_classes = 2

# # Define specific predictions and true labels for a batch size of 3
# predictions = torch.tensor([
#     [2.0, 1.0],  # Likely classified as class 0
#     [1.0, 3.0],  # Likely classified as class 1
#     [0.5, 0.5]   # Equally likely for both classes
# ])

# true_labels = torch.tensor([0, 1, 0]) # Corresponding true class labels

# # Using CrossEntropyLoss for binary classification
# criterion = nn.CrossEntropyLoss()
# criterion2 = FocalLoss(gamma=0,alpha=torch.tensor([1.0,1.0]))
# criterion4 = torch.hub.load('adeelh/pytorch-multi-class-focal-loss',
# 	model='FocalLoss',
# 	alpha=torch.tensor([1.0,1.0]),
# 	gamma=0,
# 	reduction='mean',
# 	force_reload=False)
# # Calculate the loss
# loss = criterion(predictions, true_labels)
# loss2 = criterion2(predictions, true_labels)
# loss4 = criterion4(predictions, true_labels)

# print("Loss:", loss.item())
# # Print the loss
# # print("FocalLossCustom Loss:", loss2.item())
# print("FocalLoss Loss:", loss2.item())
# print("FocalLoss2 Loss:", loss4.item())
