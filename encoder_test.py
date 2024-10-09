#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets.activation_dataset import ActivationDataset
# # %%
# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNet, self).__init__()

#         # Encoder
#         self.encoder1 = self.conv_block(in_channels, 64)
#         self.encoder2 = self.conv_block(64, 128)
#         self.encoder3 = self.conv_block(128, 256)
#         self.encoder4 = self.conv_block(256, 512)

#         # Bottleneck
#         self.bottleneck = self.conv_block(512, 1024)

#         # Decoder
#         self.decoder4 = self.conv_transpose_block(1024, 512)
#         self.decoder3 = self.conv_transpose_block(1024, 256)  # 1024 because of concatenation
#         self.decoder2 = self.conv_transpose_block(512, 128)   # 512 because of concatenation
#         self.decoder1 = self.conv_transpose_block(256, 64)

#         # Output
#         self.output = nn.Conv2d(64, out_channels, kernel_size=1)

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def conv_transpose_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # Encoder path
#         e1 = self.encoder1(x)
#         print(e1.shape)
#         e2 = self.encoder2(F.max_pool2d(e1, 2))
#         print(e2.shape)
#         e3 = self.encoder3(F.max_pool2d(e2, 2))
#         print(e3.shape)
#         e4 = self.encoder4(F.max_pool2d(e3, 2))
#         print(e4.shape)

#         # Bottleneck
#         b = self.bottleneck(F.max_pool2d(e4, 2))
#         print(b.shape)
#         # Decoder path with skip connections
#         d4 = self.decoder4(b)
#         print(d4.shape,e4.shape)
#         d4 = torch.cat((e4, d4), dim=1)
#         print(d4.shape)
#         d3 = self.decoder3(d4)
#         d3 = torch.cat((e3, d3), dim=1)
        
#         d2 = self.decoder2(d3)
#         d2 = torch.cat((e2, d2), dim=1)
        
#         d1 = self.decoder1(d2)
#         d1 = torch.cat((e1, d1), dim=1)

#         # Output layer
#         out = self.output(d1)
#         return out

# #%%
# config = {'root_dir': "/media/wmg-5gcat/Co-op Autonomy 2/Hakan/custom_dataset/nus_centerpoint_activations_filtered_objects"  ,
#       'label_file': 'nus_centerpoint_labels_filtered_objects.csv',
#       'classes': ['No Error', 'Error'],
#       'label_field': 'is_missed',
#       'layer': 0,
#       'is_multi_feature': False,
#       'is_sparse': True,
#       'extension': '.pt',
#       'name': 'nuscenes'}
# root_dir = config['root_dir'] + '/' + 'train/'
# config['root_dir'] = root_dir
# dataset = ActivationDataset(config)
# # %%
# sample,label,file = dataset[0]
# model = UNet(256, 3).cuda()
# res = model(sample.unsqueeze(0).cuda())


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.activation_dataset import ActivationDataset
from copy import deepcopy
import os
import torch.optim as optim
from tqdm.auto import tqdm  # For progress bar
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16,apply=True):
        super(SEBlock, self).__init__()
        self.apply = apply
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) if self.apply else y.expand_as(x)



class AutoencoderWithSE(nn.Module):
    def __init__(self, in_channels, latent_channels, reduction=16):
        super(AutoencoderWithSE, self).__init__()
        self.se_block = SEBlock(in_channels, reduction)
        self.encoder = nn.Conv2d(in_channels, latent_channels, kernel_size=1)
        self.decoder = nn.Conv2d(latent_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Apply Squeeze-and-Excite block before encoding
        x = self.se_block(x)
        # Encoding with 1x1 convolution
        encoded = self.encoder(x)
        # Decoding with 1x1 convolution
        decoded = self.decoder(encoded)
        return encoded, decoded

# Example usage:
import wandb
if __name__ == "__main__":
    # Assuming input is a batch of images with 3 channels (e.g., RGB)
    wandb.init(project="Introspect3D",config={},mode="online")
    act= {
    'config': {
      'root_dir': "/media/wmg-5gcat/Co-op Autonomy 2/Hakan/custom_dataset/nus_centerpoint_activations_filtered_objects_lonly/"  ,
      'label_file': 'nus_centerpoint_labels_filtered_objects_lonly.csv',
      'classes': ['No Error', 'Error'],
      'label_field': 'is_missed',
      'layer': [0,1,2],
      'is_multi_feature': True,
      'is_sparse': True,
      'extension':'.pt',
      'name': 'nuscenes'
      }
      }
    act_train = deepcopy(act)
    act_train['config']['root_dir'] = os.path.join(act['config']['root_dir'], 'train')
    act_train_dataset = ActivationDataset(act_train['config'])
    act_val = deepcopy(act)
    act_val['config']['root_dir'] = os.path.join(act['config']['root_dir'], 'val')
    act_val_dataset = ActivationDataset(act_val['config'])
    act_test = deepcopy(act)
    act_test['config']['root_dir'] = os.path.join(act['config']['root_dir'], 'test')
    act_test_dataset = ActivationDataset(act_test['config'])
    
    train_loader = torch.utils.data.DataLoader(act_train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(act_val_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(act_test_dataset, batch_size=1, shuffle=False)
    
    # Define the autoencoder with Squeeze-and-Excite
    model = AutoencoderWithSE(in_channels=256, latent_channels=1)  # Compressing to 16 channels
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10  # Number of epochs
    model.cuda()  # Move model to GPU if available
    val_loss = None
    prev_val_loss = None
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        
        # Training phase
        for data,_,_ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = data[0] # Move data to GPU if available
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            _, outputs = model(inputs)
            
            # Compute the loss
            loss = mse_loss(outputs, inputs)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate the training loss
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss over the training data
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        
        with torch.no_grad():
            for data,_,_ in val_loader:
                print(len(data))
                inputs = data[0]  # Move data to GPU if available
                
                # Forward pass
                _, outputs = model(inputs)
                
                # Compute the loss
                loss = mse_loss(outputs, inputs)
                
                # Accumulate the validation loss
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss over the validation data
        val_loss /= len(val_loader.dataset)
        if prev_val_loss == None or prev_val_loss > val_loss:
            torch.save(model.state_dict(), f"encoder_model_best.pt")
            prev_val_loss = val_loss
        wandb.log({"Train Loss":train_loss,"Val Loss":val_loss})
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # After training, evaluate on the test set if needed
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():
        for data,_,_ in test_loader:
            inputs = data[0] # Move data to GPU if available
            
            # Forward pass
            _, outputs = model(inputs)
            
            # Compute the loss
            loss = mse_loss(outputs, inputs)
            
            # Accumulate the test loss
            test_loss += loss.item() * inputs.size(0)

    # Calculate average loss over the test data
    test_loss /= len(test_loader.dataset)   
    wandb.log({"Test Loss":test_loss})
    print(f"Test Loss: {test_loss:.4f}")
    wandb.finish()
        # Forward pass

        # encoded, decoded = model(tens[0].unsqueeze(0).cpu())
        
        # print("Encoded shape:", encoded.shape)
        # print("Decoded shape:", decoded.shape)
