#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets.activation_dataset import ActivationDataset
# %%
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.decoder4 = self.conv_transpose_block(1024, 512)
        self.decoder3 = self.conv_transpose_block(1024, 256)  # 1024 because of concatenation
        self.decoder2 = self.conv_transpose_block(512, 128)   # 512 because of concatenation
        self.decoder1 = self.conv_transpose_block(256, 64)

        # Output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def conv_transpose_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        print(e1.shape)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        print(e2.shape)
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        print(e3.shape)
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        print(e4.shape)

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        print(b.shape)
        # Decoder path with skip connections
        d4 = self.decoder4(b)
        print(d4.shape,e4.shape)
        d4 = torch.cat((e4, d4), dim=1)
        print(d4.shape)
        d3 = self.decoder3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        
        d2 = self.decoder2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        
        d1 = self.decoder1(d2)
        d1 = torch.cat((e1, d1), dim=1)

        # Output layer
        out = self.output(d1)
        return out

#%%
config = {'root_dir': "/media/wmg-5gcat/Co-op Autonomy 2/Hakan/custom_dataset/nus_centerpoint_activations_filtered_objects"  ,
      'label_file': 'nus_centerpoint_labels_filtered_objects.csv',
      'classes': ['No Error', 'Error'],
      'label_field': 'is_missed',
      'layer': 0,
      'is_multi_feature': False,
      'is_sparse': True,
      'extension': '.pt',
      'name': 'nuscenes'}
root_dir = config['root_dir'] + '/' + 'train/'
config['root_dir'] = root_dir
dataset = ActivationDataset(config)
# %%
sample,label,file = dataset[0]
model = UNet(256, 3).cuda()
res = model(sample.unsqueeze(0).cuda())


# %%
