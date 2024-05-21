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
# from utils.converter import DatasetConverter
# import os
# kitti_root_path = r"/mnt/ssd2/kitti/"
# os.makedirs(os.path.join(kitti_root_path,'coco'),exist_ok=True)
# converter = DatasetConverter(root_dir=kitti_root_path,output_dir=os.path.join(kitti_root_path,'coco'),dataset_type="KITTI")
# # converter.add_category('car', 1)
# # converter.add_category('pedestrian', 2)
# # converter.add_category('truck', 3)
# # converter.add_category('bus', 4)
# # converter.add_category('train', 5)
# # converter.convert('BDD')
# converter.add_category('Car', 1)
# converter.add_category('Pedestrian', 2)
# converter.add_category('Cyclist', 3)
# converter.convert('KITTI')
# converter.split_dataset()
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from modules.custom_networks import GenericInjection
import torchmetrics
from utils.filter import EllipseFilter
from glob import glob
from utils.utils import generate_model_from_config
import os
import cv2
from mmdet3d.apis import init_model, inference_detector
activation = []
vis_early = []
#%%
def backbone_extraction_hook2(ins,inp,out):
    global vis_early
    # print(out[0].shape)
    last_output = inp[0].detach().cpu().numpy()
    vis_early.append(last_output)
def input_hook(ins,inp,out):
    global activation
    activation.append(inp[0].detach().cpu().numpy())
def backbone_extraction_hook(ins,inp,out):
    global activation
    last_output = out[0].detach().cpu().numpy()
    activation.append(last_output)
#%%
# data_folder = "/mnt/ssd2/HYY/Motorway"
data_folder = "/mnt/ssd2/HYY/Urban/"

run = "2024-04-30-12-28-22"
# labels= pd.read_csv(os.path.join(data_folder,run,"lidar_annotations_checkpoint.csv"))
data = glob(os.path.join(data_folder,run,'lidar','*.npy'))
#%%
# labels.head()

# %%
# len(data)
# %%
config = r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py'
checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'

det_model_checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
det_model_config= r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
det_model = init_model(det_model_config, det_model_checkpoint, device='cuda:0')
# print(det_model)
# exit()
# %%
int_model_config = "/mnt/ssd2/Introspect3D/configs/networks/resnet18_fcn_indv.yaml"
introspector = GenericInjection(model_config={'layer_config':int_model_config},device='cuda:0')
introspector = introspector.to('cuda:0')
#%%
# introspector.load_state_dict(torch.load("/mnt/ssd2/Introspect3D/outputs/ckpts/centerpoint_nus_activations_raw.pt",map_location='cuda:0'))
introspector.load_state_dict(torch.load("/home/yatbaz_h@WMGDS.WMG.WARWICK.AC.UK/nuscenes_inj_best.pth",map_location='cuda:0'))
# %%
print(det_model.pts_backbone._modules)
processor_hook  = det_model.pts_backbone.blocks._modules["0"].register_forward_hook(input_hook)
mid_hook = det_model.pts_backbone.blocks._modules["0"].register_forward_hook(backbone_extraction_hook)
last_hook = det_model.pts_backbone.blocks._modules["1"].register_forward_hook(backbone_extraction_hook)
# processor_hook  = det_model.pts_backbone.blocks[0].register_forward_hook(backbone_extraction_hook)
# early_hook = det_model.pts_backbone.blocks[0].register_forward_hook(backbone_extraction_hook2)
#%%

# result_df = pd.DataFrame(columns=['sample_name','label','gt','conf'])
filt = EllipseFilter(15,25,-5,1)
for sample in data:
    sample_name = os.path.basename(sample)
    # label = labels[labels['file_name']==sample_name]
    # error_label = label['error_annotation_filtered'].values[0]
    # tensor_label = torch.tensor([int(error_label)]).int().to('cuda:0')
    sample_name = sample_name.split('.')[0]
    sample_data = np.load(sample)
    sample_data = filt.filter_pointcloud(sample_data)
    five_channel_points = np.zeros((sample_data.shape[0],5))
    five_channel_points[:,:3] = sample_data
    # sample_tensor = torch.from_numpy(five_channel_points).float().to('cuda:1')
    with torch.no_grad():
        #Expand sample_data to N,5 from N,3
        
        det_results = inference_detector(det_model, five_channel_points)
        
        maxi_activation = np.max(activation[1].squeeze(),axis=0)
        plt.imshow(maxi_activation)
        plt.show()
        # print(maxi_activation.shape)
        # tensor_activation = [torch.from_numpy(act.squeeze()).float().to('cuda:0').unsqueeze(0) for act in activation]

        # # sample_activation = torch.from_numpy(sample_activation).to('cuda:0')
        # output = introspector(tensor_activation)
        # softmaxed_output = torch.nn.functional.softmax(output,dim=1)
        # softmaxed_pred = torch.argmax(softmaxed_output,dim=1)
        
        # temp_df = pd.DataFrame({'sample_name':sample_name,'label':softmaxed_pred.item(),'gt':int(error_label),'conf':softmaxed_output[0,softmaxed_pred.item()].item()},index=[0])
        # result_df = pd.concat([result_df,temp_df],ignore_index=True)
    # img = vis_early[0]
    # print(img.shape)
    # img_max_channelwise = np.max(img,axis=1)
    # cv2.imshow('img',img_max_channelwise)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    activation = []
processor_hook.remove()
mid_hook.remove()
last_hook.remove()
# result_df.to_csv(os.path.join(data_folder,run,'introspection_results2.csv'),index=False)

# %%
