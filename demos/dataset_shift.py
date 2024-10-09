
from base_classes.base import Operator
from utils.factories import DatasetFactory
from utils.utils import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from definitions import ROOT_DIR
from utils.process import *
import wandb
import torch
from modules.graph_networks import GCN
from modules.custom_networks import CustomModel
from pprint import pprint
import random
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AveragePrecision, AUROC, ConfusionMatrix, StatScores

def calculate_torchmetrics(loader, model, processor, device="cuda:1",task = 'multiclass'):
    num_classes = 2
    metric_collection = MetricCollection([
        ConfusionMatrix(num_classes=num_classes,task=task),
        # Accuracy(task=task,num_classes=num_classes,average='none'),
        # Precision(pos_label=1,task=task,num_classes=num_classes,average='none'),
        Recall(pos_label=1,task=task,num_classes=num_classes,average='none'),
        F1Score(task=task,num_classes=num_classes,average='none'),
        AUROC(task=task,num_classes=num_classes,pos_label=1,average='none'),
        # StatScores(num_classes=num_classes,task=task,average='none'),
        AveragePrecision(num_classes=num_classes,task=task,average='none'),
    ]).to(device)

    model.eval()
    model.to(device)

    with tqdm(loader, total=len(loader), desc="Calculating metrics") as pbar:
        for data, target, _ in loader:
            target = target.squeeze()
            target = target.to(device)
            data = processor.process(activation=data)
            data = torch.from_numpy(data).to(device).float()
            output = model(data)
            output.to(device)
            target=target.to(device)
            # print(device)
            metric_collection.update(output, target)
            pbar.update(1)

    metrics = {name: metric.compute() for name, metric in metric_collection.items()}
    pprint(metrics)
    return metrics


# def calculate_torchmetrics(pred,target,mode = 'train',task = 'multiclass',iteration=1):
#     num_classes = 2
#     print(pred.shape,target.shape)
#     pred = torch.tensor(pred).squeeze()
#     target = torch.tensor(target,dtype=torch.int64).squeeze()
    
#     metric_collection = MetricCollection([
#         ConfusionMatrix(num_classes=num_classes,task=task),
#         # Accuracy(task=task,num_classes=num_classes,average='none'),
#         # Precision(pos_label=1,task=task,num_classes=num_classes,average='none'),
#         Recall(pos_label=1,task=task,num_classes=num_classes,average='none'),
#         F1Score(task=task,num_classes=num_classes,average='none'),
#         AUROC(task=task,num_classes=num_classes,pos_label=1,average='none'),
#         # StatScores(num_classes=num_classes,task=task,average='none'),
#         AveragePrecision(num_classes=num_classes,task=task,average='none'),
#     ])
#     metric_collection.to("cuda:1")
#     metrics = metric_collection(pred,target)
#     from pprint import pprint
#     pprint(metrics)
#     #This is messy but to try
#     return metrics
# def calculate_torchmetrics(pred,target,mode = 'train',task = 'multiclass',iteration=1):
#     num_classes = 2
#     print(pred.shape,target.shape)
#     pred = torch.tensor(pred).squeeze()
#     target = torch.tensor(target,dtype=torch.int64).squeeze()
    
#     metric_collection = MetricCollection([
#         ConfusionMatrix(num_classes=num_classes,task=task),
#         # Accuracy(task=task,num_classes=num_classes,average='none'),
#         # Precision(pos_label=1,task=task,num_classes=num_classes,average='none'),
#         Recall(pos_label=1,task=task,num_classes=num_classes,average='none'),
#         F1Score(task=task,num_classes=num_classes,average='none'),
#         AUROC(task=task,num_classes=num_classes,pos_label=1,average='none'),
#         # StatScores(num_classes=num_classes,task=task,average='none'),
#         AveragePrecision(num_classes=num_classes,task=task,average='none'),
#     ])
#     metric_collection.to("cuda:1")
#     metrics = metric_collection(pred,target)
#     from pprint import pprint
#     pprint(metrics)
#     #This is messy but to try
#     return metrics

if __name__ == "__main__":
    config = "configs/wmg-pc/yolov8_kitti_shift.yaml"
    # model_save= "bdd_none_resnet18_fcn_padash2.pth"
    model_save= "bdd_none_resnet18_fcn_pad4.pth"
    conf = Config(config)
    int_conf = conf.introspection
    method_info = int_conf['method']
    proceesor = eval(method_info['processing']['method']).value(**method_info['processing']['params'])
    dataset = DatasetFactory().get(**int_conf['dataset'])
    model = generate_model_from_config(int_conf['method']['model'])
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR,model_save)))
    all_preds = torch.tensor([]).to("cpu",dtype=torch.float32) 
    all_labels = torch.tensor([]).to("cpu",dtype=torch.long)
    print(dataset[0])
    model.eval()
    model.to(int_conf['device'])
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    metrics = calculate_torchmetrics(loader, model, proceesor, device=int_conf['device'])

    # with tqdm(dataset, total=len(dataset), desc="Calculating metrics") as pbar:
    #     for d in loader:
    #         data,target,_ = d
    #         target = target.to('cpu')
    #         # print(data.shape,target.shape)
    #         data = proceesor.process(activation=data)
    #         data = torch.from_numpy(data).to('cuda:1')
    #         data = data.float()
    #         # print("-",data.shape)
    #         output = model(data)
    #         output = output.to('cpu')
    #         # print("--",output.shape)
    #         all_preds = torch.cat(
    #                     (all_preds, output),dim=0
    #                 )
    #         all_labels = torch.cat(
    #                 (all_labels, target),dim=0
    #             )
    #         pbar.update(1)
    # calculate_torchmetrics(all_preds,all_labels,mode='all',task=method_info['task'])
    