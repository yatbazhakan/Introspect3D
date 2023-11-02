
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
from pprint import pprint
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AveragePrecision, AUROC, ConfusionMatrix, StatScores
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
        self.model_save_to = os.path.join(ROOT_DIR,self.method_info['save_path'],self.method_info['save_name'])
        if self.method_info['processing']['active']:
            print(self.method_info['processing']['method'])
            self.proceesor = eval(self.method_info['processing']['method']).value(**self.method_info['processing']['params'])
        else:
            self.proceesor = None

    def get_dataloader(self):
        if self.split:
            train_loader = DataLoader(self.train_dataset, **self.config['dataloader']['train'])
            test_loader = DataLoader(self.test_dataset, **self.config['dataloader']['test'])
            self.train_loader = train_loader
            self.test_loader = test_loader
        else: # THis will give me flexibility for dataset shift case
            loader = DataLoader(self.dataset, **self.config['dataloader']['all'])
            self.test_loader = loader

    def train_test_split(self):
        indices = list(range(len(self.dataset)))
        all_labels= self.dataset.get_all_labels()
        train_indices, test_indices = train_test_split(indices, test_size=self.method_info['train_test_split'],stratify=all_labels)
        self.train_dataset = Subset(self.dataset, train_indices)
        self.test_dataset = Subset(self.dataset, test_indices)
        self.split=True
        if self.verbose:
            #Provide the class distribution overall
            values,counts = np.unique(all_labels,return_counts=True)
            print("Class distribution:",dict(zip(values,counts)))

    def initialize_learning_parameters(self):
        self.device = self.config['device']

        self.split = False
        if self.method_info['train_test_split'] != None:
            self.train_test_split()
        self.get_dataloader()

        self.optimizer = generate_optimizer_from_config(self.method_info['optimizer'],self.model)
        self.criterion = generate_criterion_from_config(self.method_info['criterion'])
        self.scheduler = generate_scheduler_from_config(self.method_info['scheduler'],self.optimizer)
        self.epochs = self.method_info['epochs']
        self.log_interval = self.method_info['log_interval']
        self.save_interval = self.method_info['save_interval']
    def train_epoch(self,epoch):
        epoch_loss = 0
        if self.verbose:
            print("Epoch:",epoch)
        pbar = tqdm(total=len(self.train_loader),leave=False)
        for batch_idx, (data, target, name) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            if(self.proceesor != None):
                data = self.proceesor.process(activation=data)
                data = torch.from_numpy(data).to(self.device)
            output = self.model(data)
            output= output.float()
            target = target.long()
            loss = self.criterion(output, target.squeeze())
            loss.backward()
            self.optimizer.step()
            self.total_loss += loss.item()
            epoch_loss += loss.item()
            pbar.update(1)
        #TODO: check if this division is correct way to do so
        return epoch_loss/len(self.train_loader.dataset)

    def train(self):
        #MNot very OOP of me but this might improve earlier waiting times
        self.dataset = DatasetFactory().get(**self.config['dataset'])
        self.model = generate_model_from_config(self.method_info['model'])
        
        

        early_stop_threshold = self.method_info['early_stop']
        no_improvement_count = 0

        self.model = self.model.to(self.device)
        
        self.initialize_learning_parameters()
        self.criterion.to(self.device)
        self.total_loss = np.inf
        previous_loss = np.inf
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(epoch)
            if self.verbose:
                print("Epoch loss:",epoch_loss)
            wandb.log({'epoch_loss':epoch_loss})
            if epoch_loss < previous_loss:
                previous_loss = epoch_loss
                no_improvement_count = 0
                if self.verbose:
                    print("Saving model")
                torch.save(self.model.state_dict(), self.model_save_to)
                wandb.save(self.method_info['save_path'])
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_threshold:
                    if self.verbose:
                        print("Early stopping")
                    break
            if self.method_info['sanity_check']:
                self.evaluate(loader=self.train_loader,epoch=epoch)
                self.evaluate(loader=self.test_loader,epoch=epoch)
        

    def evaluate(self,loader=None,epoch=None):
        if loader == None:
            if self.verbose:
                print("No loader provided, Evaluation started for test set")
            loader = self.test_loader
        self.model.eval()
        test_loss = 0
        all_preds = torch.tensor([]).to(self.config['device'],dtype=torch.float32) 
        all_labels = torch.tensor([]).to(self.config['device'],dtype=torch.long)
        with torch.no_grad():
            for data, target, name in loader:

                data, target = data.to(self.device), target.to(self.device)
                if(self.proceesor != None):
                    data = self.proceesor.process(activation=data)
                    data = torch.from_numpy(data).to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target.squeeze(1)).item()
                all_preds = torch.cat(
                    (all_preds, output),dim=0
                )
                all_labels = torch.cat(
                        (all_labels, target),dim=0
                    )
        test_loss /= len(loader.dataset)       
        
        if loader == self.test_loader:
            wandb.log({'test_loss':test_loss})
            self.calculate_torchmetrics(all_preds,all_labels,mode='test')
        elif loader == self.train_loader:
            wandb.log({'train_loss':test_loss})
            self.calculate_torchmetrics(all_preds,all_labels,mode='train')
    def seprate_multiclass_metrics(self,metric_name):
        multi_class_metric = self.metrics[metric_name]
        try:
            positive_metric = multi_class_metric[1].cpu().numpy()
            negative_metric = multi_class_metric[0].cpu().numpy()
        except:
            print(metric_name,multi_class_metric)
        return positive_metric,negative_metric
    def log_metrics(self,mode):
        #Here I need to separate metrics for each class and log them in wandb
        cm = self.metrics['MulticlassConfusionMatrix']
        cm = cm.cpu().numpy()
        cm = cm.astype(int)
        tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        wandb.log({f'{mode}_tp':tp,f'{mode}_fp':fp,f'{mode}_fn':fn,f'{mode}_tn':tn})

        for metric_name in self.metrics.keys():
            if metric_name == 'MulticlassConfusionMatrix':
                continue
            positive_metric,negative_metric = self.seprate_multiclass_metrics(metric_name)
            # print(positive_metric)
            try:
                for i in range(positive_metric.shape[0]):
                    wandb.log({f'{mode}_{metric_name}_positive_{i}':positive_metric[i]})
                for i in range(negative_metric.shape[0]):
                    wandb.log({f'{mode}_{metric_name}_negative_{i}':negative_metric[i]})
            except:
                wandb.log({f'{mode}_{metric_name}_positive':positive_metric})
                wandb.log({f'{mode}_{metric_name}_negative':negative_metric})


    def calculate_torchmetrics(self,pred,target,mode = 'train'):
        num_classes = 2
        task = 'multiclass'
        pred = torch.tensor(pred).squeeze()
        target = torch.tensor(target,dtype=torch.int64).squeeze()
        metric_collection = MetricCollection([
            ConfusionMatrix(num_classes=num_classes,task=task),
            Accuracy(task=task,num_classes=num_classes,average='none'),
            Precision(pos_label=1,task=task,num_classes=num_classes,average='none'),
            Recall(pos_label=1,task=task,num_classes=num_classes,average='none'),
            F1Score(task=task,num_classes=num_classes,average='none'),
            AUROC(task=task,num_classes=num_classes,pos_label=1,average='none'),
            StatScores(num_classes=num_classes,task=task,average='none'),
            AveragePrecision(num_classes=num_classes,task=task,average='none'),
        ])
        metric_collection.to(self.device)
        self.metrics = metric_collection(pred,target)
        from pprint import pprint
        # pprint(self.metrics)
        #This is messy but to try
        self.log_metrics(mode)
        

    def execute(self, **kwargs):
        if self.is_sweep:
            sweep_configuration = self.wandb['sweep_configuration']
            # sweep_id = wandb.sweep(sweep=sweep_configuration, project='introspectionBase')
            if self.verbose:
                print("Sweep id:",None)
                # print("="*100,"\n",wandb.config,"\n","="*100)
            # wandb.agent(sweep_id, function=self.train,count=1)
        else:
            #Some management will be needed here
            wandb.init(project=self.wandb['project'],config=self.config, entity=self.wandb['entity'],mode=self.wandb['mode'],name=self.wandb['name'])
            if self.config['operation']['type'] == "train":
                self.train()
            elif self.config['operation']['type'] == "evaluate":
                self.evaluate()
            else:
                self.train()
                self.evaluate()