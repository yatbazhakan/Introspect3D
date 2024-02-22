
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
from modules.custom_networks import CustomModel, EarlyFusionAdaptive
from pprint import pprint
import random
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AveragePrecision, AUROC, ConfusionMatrix, StatScores
import os
class IntrospectionOperator(Operator):
    def __init__(self,config) -> None:
        super().__init__()
        
        self.config = config.introspection
        self.wandb = self.config['wandb']
        self.verbose = self.config['verbose']
        self.method_info = self.config['method']
        self.is_sweep = self.wandb['is_sweep']
        self.device = self.config['device']
        os.makedirs(os.path.join(ROOT_DIR,self.method_info['save_path']),exist_ok=True)
        if self.method_info['processing']['active']:
            print(self.method_info['processing']['method'])
            self.proceesor = eval(self.method_info['processing']['method']).value(**self.method_info['processing']['params'])
        else:
            self.proceesor = None
    def name_builder(self):

        dataset = self.config['dataset']['config']['name'].lower()
        network = os.path.splitext(os.path.basename(self.method_info['model']['layer_config']))[0]
        procesing = self.method_info['processing']['method'].split(".")[-1].lower()
        filtering = self.config ['filtering'].lower()
        self.method_info['save_name'] = f"{dataset}_{filtering}_{network}_{procesing}"
    def get_dataloader(self):
        if self.split:
            train_loader = DataLoader(self.train_dataset, **self.config['dataloader']['train'])
            test_loader = DataLoader(self.test_dataset, **self.config['dataloader']['test'])
            val_loader = DataLoader(self.val_dataset, **self.config['dataloader']['test'])
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.val_loader = val_loader
        else: # THis will give me flexibility for dataset shift case
            loader = DataLoader(self.dataset, **self.config['dataloader']['all'])
            self.test_loader = loader

    def train_test_split(self):
        indices = list(range(len(self.dataset)))
        all_labels= self.dataset.get_all_labels()
        validation = self.method_info['cross_validation']
        random_state = random.randint(0,2048) if validation['type'] == "montecarlo" and  not self.is_sweep else 1024
        train_indices, test_indices = train_test_split(indices, test_size=validation['train_test_split'],stratify=all_labels,random_state=random_state)
        self.train_dataset = Subset(self.dataset, train_indices)
        after_val_train_indices, val_indices = train_test_split(train_indices, test_size=validation['validation_split'],stratify=all_labels[train_indices],random_state=random_state)
        values,counts = np.unique(all_labels,return_counts=True)
        class_dist=  dict(zip(values,counts))
        c = 1e-3
        class_weights = [len(all_labels)/float(count) for cls, count in class_dist.items()]
        if validation['balanced']:
            train_labels_after_val = [all_labels[i] for i in after_val_train_indices]
            class_counts = np.bincount(train_labels_after_val)
            min_class_count = np.min(class_counts)
            print("Min class count:",min_class_count,"Class counts:",class_counts)
            balanced_train_indices = []
            for cls in np.unique(train_labels_after_val):
                cls_indices = [i for i, label in zip(after_val_train_indices, train_labels_after_val) if label == cls]
                balanced_cls_indices = np.random.choice(cls_indices, min_class_count, replace=False)
                balanced_train_indices.extend(balanced_cls_indices)
            after_val_train_indices = balanced_train_indices
            balanced_train_labels = [all_labels[i] for i in balanced_train_indices]
            values, counts = np.unique(balanced_train_labels, return_counts=True)
            new_class_dist = dict(zip(values, counts))
            total_samples = len(balanced_train_indices)
            class_weights = [ total_samples / (len(values) * count) for cls, count in new_class_dist.items()]
            class_dist = new_class_dist
        
        
        
        self.train_dataset = Subset(self.dataset, after_val_train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)


        self.split=True
        #Provide the class distribution overall

        # class_weights = [1 / (np.log(c + count)) for cls, count in class_dist.items()]
        self.method_info['criterion']['params']['weight'] = torch.FloatTensor(class_weights).to(self.config['device'])

        # if self.method_info['criterion']['type'] == 'CrossEntropyLoss':
        #     # class_weights = [float(i)/sum(class_weights) for i in class_weights]
        #     self.method_info['criterion']['params']['weight'] = torch.FloatTensor(class_weights).to(self.config['device'])
        # elif self.method_info['criterion']['type'].startswith("FocalLoss"):
        #     #Getting second element of class weights since it is the error class (positive class is 1)
        #     #Scale weights between 0 and 1 using sum
        #     # class_weights = [float(i)/sum(class_weights) for i in class_weights]
        #     if "Custom" not in self.method_info['criterion']['type']:
        #         self.method_info['criterion']['params']['alpha'] = torch.tensor(class_weights[1]).to(self.config['device'])
        #     else:
        #         self.method_info['criterion']['params']['weight'] = torch.tensor(class_weights).to(self.config['device'])
        if self.verbose:

            print("Class distribution:",class_dist)
            print("Class weights:",class_weights)

    def initialize_learning_parameters(self):
        self.device = self.config['device']
        #DEBUG*  Possible that we just need to get the values or names, and get the built info from regular config
        optimizer_info = self.method_info['optimizer']
        criterion_info = self.method_info['criterion']
        self.split = False
        
        if self.method_info['cross_validation']['type'] != None:
            self.train_test_split()
        self.get_dataloader()

        self.optimizer = generate_optimizer_from_config(optimizer_info,self.model)
        self.criterion = generate_criterion_from_config(criterion_info)
        self.scheduler = generate_scheduler_from_config(self.method_info['scheduler'],self.optimizer)
        self.epochs = self.method_info['epochs']
        self.log_interval = self.method_info['log_interval']
        self.save_interval = self.method_info['save_interval']
        
    def train_epoch(self,epoch):
        epoch_loss = 0
        if self.verbose:
            print("Epoch:",epoch)
        pbar = tqdm(total=len(self.train_loader),leave=False)
        pbar.set_description(f'Training at Epoch {epoch} with learning rate {self.optimizer.param_groups[0]["lr"]:.2E} and No improvement count {self.scheduler.num_bad_epochs}')
        for batch_idx, (data, target, name) in enumerate(self.train_loader):
            
            # print(data.shape)
            self.optimizer.zero_grad()
            if(self.proceesor != None and 
               "GAP" not in self.method_info['processing']['method'] and
               "MULTI" not in self.method_info['processing']['method'] and
               "EFS" not in self.method_info['processing']['method']):
                data = self.proceesor.process(activation=data,stack=self.method_info['processing']['stack'])
                # print("Processed data",data.shape)
                if type(data) == np.ndarray:
                    data = torch.from_numpy(data).to(self.device)
                data = data.float()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
            elif "GAP" in self.method_info['processing']['method']:
                data, target = data.to(self.device), target.to(self.device)
                batched_data = []
                edge_indexes,node_features = self.proceesor.process(activation=data)
                from torch_geometric.data import Data,Batch
                for i in range(len(node_features)):
                    features = torch.from_numpy(node_features[i])
                    features = features.float()
                    indexes = torch.from_numpy(edge_indexes[i])
                    indexes = indexes.long()
                    data = Data(x=features, edge_index=indexes).to(self.device)
                    batched_data.append(data)
                
                # batched_data = torch.from_numpy(np.array(batched_data))
                batch = Batch.from_data_list(batched_data)
                data = batch.to(self.device)#batched_data
                # data = batched_data
                # print(data)
                # data = data.to('cpu')
                self.model = self.model.to(self.device)
                output = self.model(data)
                # print(output.shape)
            elif "MULTI" in self.method_info['processing']['method'] or "EFS" in self.method_info['processing']['method']:
                target = target.to(self.device)
                output = self.model(data)
            # elif "EFS" in self.method_info['processing']['method']:
            #     data, target = data.to(self.device), target.to(self.device)
            #     output = self.model(data)
            if self.method_info['criterion']['type'] == 'BCEWithLogitsLoss':
                # print(target.shape, output.shape)
                loss = self.criterion(output, target.float())
            else:
                output= output.float()
                target = target.long()
                loss = self.criterion(output, target.squeeze())

            loss.backward()
            self.optimizer.step()
            self.total_loss += loss.item()
            # print(loss.item())
            epoch_loss += loss.item()
            pbar.update(1)
            clear_memory()
        #TODO: check if this division is correct way to do so
        return epoch_loss
    def update_config_from_wandb(self,conf):
        wandb.log({"custom_config":self.config})
        self.method_info['model']['layer_config'] = wandb.config.model_yaml
        self.method_info['optimizer']['type'] = wandb.config.optimizer
        self.method_info['optimizer']["params"]["lr"] = wandb.config.lr
        self.method_info['criterion']['type'] = wandb.config.criterion
        self.config['dataloader']['train']['batch_size'] = wandb.config.batch_size
    def train(self,iteration=1):
        #MNot very OOP of me but this might improve earlier waiting times, may need a wrapper around wandb.config to mapit to the actual config?
        if self.is_sweep:
            print("Sweep configuration")
            run = wandb.init(project=self.wandb['project'], entity=self.wandb['entity'],mode=self.wandb['mode'],name=self.wandb['name'])
            run_config = run.config
            print("="*100,"\n",run_config,run_config.keys(),"\n","="*100)
            self.update_config_from_wandb(run_config)
        print("Name Builder")
        self.name_builder()

        self.model_save_to = os.path.join(ROOT_DIR,self.method_info['save_path'],self.method_info['save_name']+f"{iteration}.pth")
        self.dataset = DatasetFactory().get(**self.config['dataset'])
        model_info = self.method_info['model']            
        if "GAP" in self.method_info['processing']['method']:
            self.model = GCN(model_info)
            print(self.model)
        elif "MULTI" in self.method_info['processing']['method']:
            self.model = CustomModel(model_info,device=self.device)
        elif "EFS" in self.method_info['processing']['method']:
            print("LOADING EFS MODEL")
            self.model = EarlyFusionAdaptive(model_info,device=self.device)
        else:
            self.model = generate_model_from_config(model_info)
        print("Model loaded")
        early_stop_threshold = self.method_info['early_stop']
        no_improvement_count = 0

        self.model = self.model.to(self.device)
        print("Learning parameters initialized")
        self.initialize_learning_parameters()
        self.criterion.to(self.device)
        self.total_loss = np.inf
        previous_val_loss = np.inf
        val_loss = np.inf
        print("Training started")
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(epoch)
            if epoch_loss < 0.001:
                print("Train loss is converged")
                break
            if self.verbose:
                print("Epoch loss:",epoch_loss)
            wandb.log({'epoch_loss':epoch_loss})
            
            if self.method_info['sanity_check']:
                self.evaluate(loader=self.train_loader,epoch=epoch)
            val_loss = self.evaluate(loader=self.val_loader,epoch=epoch)
                        
            if val_loss < previous_val_loss:
                previous_val_loss = val_loss
                no_improvement_count = 0

                if self.verbose:
                    print("Saving model")
                torch.save(self.model.state_dict(), self.model_save_to)
                wandb.save( self.model_save_to)
            else:
                if self.verbose:
                    print("No improvement")
                no_improvement_count += 1
                self.scheduler.step(val_loss)
                if no_improvement_count >= early_stop_threshold:
                    if self.verbose:
                        print("Early stopping")
                    break


    def evaluate(self,iteration=1,loader=None,epoch=None):
        if loader == None:
            
            loader = self.test_loader
            self.model.load_state_dict(torch.load(self.model_save_to))
            if self.verbose:
                print("No loader provided, Evaluation started for test set")
                print("Model loaded from:",self.model_save_to)

        self.model.eval()
        test_loss = 0
        all_preds = torch.tensor([]).to(self.config['device'],dtype=torch.float32) 
        all_labels = torch.tensor([]).to(self.config['device'],dtype=torch.long)
        with torch.no_grad():
            with tqdm(total=len(loader),leave=False) as pbar:
                pbar.set_description(f'Evaluating at Epoch {epoch}')
                pbar.refresh()
                for data, target, name in loader:
                    if(self.proceesor != None and "GAP" not in self.method_info['processing']['method'] and
                       "MULTI" not in self.method_info['processing']['method']
                       and "EFS" not in self.method_info['processing']['method']):
                    
                        data = self.proceesor.process(activation=data,stack=self.method_info['processing']['stack'])
                        # print(type(data))
                        data = torch.from_numpy(data).to(self.device) if isinstance(data,np.ndarray) else data.to(self.device)
                        data, target = data.to(self.device), target.to(self.device)

                        data = data.float()
                    elif "GAP" in self.method_info['processing']['method']:
                        data, target = data.to(self.device), target.to(self.device)
                        batched_data = []
                        edge_indexes,node_features = self.proceesor.process(activation=data)
                        from torch_geometric.data import Data,Batch
                        for i in range(len(node_features)):
                            features = torch.from_numpy(node_features[i])
                            features = features.float()
                            indexes = torch.from_numpy(edge_indexes[i])
                            indexes = indexes.long()
                            data = Data(x=features, edge_index=indexes).to(self.device)
                            batched_data.append(data)
                        batch = Batch.from_data_list(batched_data)
                        data = batch.to(self.device)#batched_data
                        # data = batched_data
                        # print(data)
                        # data = data.to('cpu')
                        self.model = self.model.to(self.device)
                    elif "MULTI" in self.method_info['processing']['method'] or "EFS" in self.method_info['processing']['method']:
                        target = target.to(self.device)
                    output = self.model(data)

                    if self.method_info['criterion']['type'] == 'BCEWithLogitsLoss':
                        test_loss = self.criterion(output, target.float()).item()
                    else:
                        test_loss += self.criterion(output, target.squeeze(1)).item()
                    all_preds = torch.cat(
                        (all_preds, output),dim=0
                    )
                    all_labels = torch.cat(
                            (all_labels, target),dim=0
                        )
                    pbar.update(1)
                    clear_memory()
        # test_loss /= len(loader.dataset)       
        
        if loader == self.test_loader:
            wandb.log({f'test_loss_{iteration}':test_loss})
            self.calculate_torchmetrics(all_preds,all_labels,mode='test',task=self.method_info['task'])
        elif loader == self.train_loader:
            wandb.log({f'train_loss_{iteration}':test_loss})
            self.calculate_torchmetrics(all_preds,all_labels,mode='train',task=self.method_info['task'])
        else :
            wandb.log({f'val_loss_{iteration}':test_loss})
            self.calculate_torchmetrics(all_preds,all_labels,mode='val',task=self.method_info['task'])
        return test_loss
    def seprate_multiclass_metrics(self,metric_name):
        multi_class_metric = self.metrics[metric_name]
        try:
            positive_metric = multi_class_metric[1].cpu().numpy()
            negative_metric = multi_class_metric[0].cpu().numpy()
        except:
            print(metric_name,multi_class_metric)
        return positive_metric,negative_metric
    def log_metrics(self,mode,task,iteration=1):
        #Here I need to separate metrics for each class and log them in wandb
        for metric_name in self.metrics.keys():
            if 'ConfusionMatrix' in metric_name:
                cm = self.metrics[metric_name]
                # wandb_table = wandb.Table(data=cm.cpu().numpy().tolist(), columns=["Predicted Safe", "Predicted Error"])
                # wandb.log({f'{mode}_confusion_matrix_{iteration}':wandb_table})
                cm = cm.cpu().numpy()
                cm = cm.astype(int)
                tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
                wandb.log({f'{mode}_tp':tp,f'{mode}_fp':fp,f'{mode}_fn':fn,f'{mode}_tn':tn})
            else:
                if task == 'multiclass':
                    positive_metric,negative_metric = self.seprate_multiclass_metrics(metric_name)
                    # print(positive_metric)
                    try:
                        for i in range(positive_metric.shape[0]):
                            wandb.log({f'{mode}_{metric_name}_positive_{i}_{iteration}':positive_metric[i]})
                        for i in range(negative_metric.shape[0]):
                            wandb.log({f'{mode}_{metric_name}_negative_{i}_{iteration}':negative_metric[i]})
                    except:
                        wandb.log({f'{mode}_{metric_name}_positive_{iteration}':positive_metric})
                        wandb.log({f'{mode}_{metric_name}_negative_{iteration}':negative_metric})
                else:
                    wandb.log({f'{mode}_{metric_name}_{iteration}':self.metrics[metric_name].cpu().numpy()})


    def calculate_torchmetrics(self,pred,target,mode = 'train',task = 'multiclass',iteration=1):
        num_classes = 2
        pred = torch.tensor(pred).squeeze()
        target = torch.tensor(target,dtype=torch.int64).squeeze()
        
        metric_collection = MetricCollection([
            ConfusionMatrix(num_classes=num_classes,task=task),
            # Accuracy(task=task,num_classes=num_classes,average='none'),
            # Precision(pos_label=1,task=task,num_classes=num_classes,average='none'),
            Recall(pos_label=1,task=task,num_classes=num_classes,average='none'),
            F1Score(task=task,num_classes=num_classes,average='none'),
            AUROC(task=task,num_classes=num_classes,pos_label=1,average='none'),
            # StatScores(num_classes=num_classes,task=task,average='none'),
            AveragePrecision(num_classes=num_classes,task=task,average='none'),
        ])
        metric_collection.to(self.device)
        self.metrics = metric_collection(pred,target)
        from pprint import pprint
        # pprint(self.metrics)
        #This is messy but to try
        self.log_metrics(mode,task,iteration)
        
    def train_sweep(self): #Basic wrapper for wandb sweep with montecarlo cross validation
        print("Wrapper initialized")
        for i in range(self.method_info['cross_validation']['iteration']):
            print("Iteration:",i)
            self.train(i)
            self.evaluate(i)
        
    def execute(self, **kwargs):
        
       
        try:
            if self.is_sweep:
                os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "1"
                os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"
                os.environ['WANDB_AGENT_FLAPPING_MAX_FAILURES'] = '1'

                sweep_configuration = self.wandb['sweep_configuration']
                sweep_id = wandb.sweep(sweep=sweep_configuration, project=self.wandb['project'])
                if self.verbose:
                    print("Sweep id:",None)
                    print("="*100,"\n",wandb.config,"\n","="*100)
                wandb.agent(sweep_id, function=self.train_sweep)
           
                
            else:
                #Some management will be needed here
                wandb.init(project=self.wandb['project'],config=self.config, entity=self.wandb['entity'],mode=self.wandb['mode'],name=self.wandb['name'])
                
                if self.config['operation']['type'] == "train":
                    for i in range(self.method_info['cross_validation']['iteration']):
                        self.train(i)
                elif self.config['operation']['type'] == "evaluate":
                    self.dataset = DatasetFactory().get(**self.config['dataset'])
                    self.model.to(self.device)
                    self.initialize_learning_parameters()
                    self.evaluate()
                else:
                    for i in range(self.method_info['cross_validation']['iteration']):
                        self.train(i)
                        self.evaluate(i)

        except Exception as e:
            clear_memory()
            exit(e)