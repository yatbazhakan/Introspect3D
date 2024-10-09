import torch
def generate_optimizer_from_config(config,model):
    optimizer_type = config['type']
    optimizer_params = config['params']
    return eval(f"{optimizer_type}(model.parameters(),**optimizer_params)")

def generate_scheduler_from_config(config,optimizer):
    scheduler_type = config['type']
    scheduler_params = config['params']
    return eval(f"{scheduler_type}(optimizer,**scheduler_params)")

def generate_criterion_from_config(config,**kwargs):
    print(config)
    loss_type = config['type']
    loss_params = config['params']
    if "weight" in loss_params.keys():
        loss_params['weight'] = torch.tensor(loss_params['weight'],dtype=torch.float32)
    return eval(f"{loss_type}(**loss_params)")
