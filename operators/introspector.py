
from base_classes.base import Operator
from utils.factories import DatasetFactory
from utils.utils import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import torch
class IntrospectionOperator(Operator):
    def __init__(self,config) -> None:
        super().__init__()
        self.wandb = self.config['wandb']
        self.config = config.introspection
        self.verbose = self.config['verbose']
        self.method_info = self.config['method']
        self.is_sweep = self.wandb['is_sweep']

    def get_dataloader(self):
        if self.split:
            train_loader = DataLoader(self.train_dataset, **self.config['dataloader']['train'])
            test_loader = DataLoader(self.test_dataset, **self.config['dataloader']['test'])
            self.train_loader = train_loader
            self.test_loader = test_loader
        else:
            loader = DataLoader(self.dataset, **self.config['dataloader']['all'])
            self.test_loader = loader

    def train_test_split(self):
        indices = list(range(len(self.dataset)))
        all_labels= self.dataset.get_all_labels()
        train_indices, test_indices = train_test_split(indices, self.method_info['train_test_split'],stratify=all_labels)
        self.train_dataset = Subset(self.dataset, train_indices)
        self.test_dataset = Subset(self.dataset, test_indices)
        self.split=True
    
    def initialize_learning_parameters(self):
        self.device = self.config['device']

        self.split = False
        if self.method_info['train_test_split']:
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
        for batch_idx, (data, target, name) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.total_loss += loss.item()
            epoch_loss += loss.item()
        #TODO: check if this division is correct way to do so
        return epoch_loss/len(self.train_loader.dataset)

    def train(self):
        #MNot very OOP of me but this might improve earlier waiting times
        self.dataset = DatasetFactory().get(**self.config['dataset'])
        self.model = generate_model_from_config(self.method_info['model'])
        self.initialize_learning_parameters()

        self.model.to(self.device)
        self.model.train()
        self.total_loss = np.inf
        previous_loss = np.inf
        pbar = tqdm(total=self.epochs)
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(epoch)
            if self.verbose:
                pbar.update(1)
            wandb.log({'epoch_loss':epoch_loss})
            if epoch_loss < previous_loss:
                previous_loss = epoch_loss
                if self.verbose:
                    print("Saving model")
                torch.save(self.model.state_dict(), self.method_info['save_path'])


        

    def execute(self, **kwargs):
        if self.is_sweep:
            sweep_configuration = self.wandb['sweep_configuration']
            sweep_id = wandb.sweep(sweep=sweep_configuration, project='introspectionBase')
            if self.verbose:
                print("="*100,"\n",wandb.config,"\n","="*100)
            wandb.agent(sweep_id, function=self.train,count=1)
        else:
            #Some management will be needed here
            wandb.init(project=self.wandb['project'],config=self.config, entity=self.wandb['entity'],mode=self.wandb['mode'])
            self.train()
