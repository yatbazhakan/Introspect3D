import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ActivationMapTokenizer(nn.Module):
    def __init__(self, C, H, W, embed_dim):
        super(ActivationMapTokenizer, self).__init__()
        self.C = C
        self.H = H
        self.W = W
        self.embed_dim = embed_dim
        
        # Linear projection of flattened tokens to embed_dim
        self.projection = nn.Linear(H * W, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)  # Adding LayerNorm for stabilization

    def forward(self, x):
        # x: input activation map with shape (batch_size, C, H, W)
        batch_size = x.size(0)
        # Flatten each channel to create tokens of shape (batch_size, C, H*W)
        x = x.view(batch_size, self.C, -1)
        # Project flattened tokens to the desired embedding dimension
        x = self.projection(x)  # Now shape (batch_size, C, embed_dim)
        x = self.norm(x)  # Apply normalization
        return x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, C, embed_dim):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        
        # Create a buffer for the positional encodings
        position = torch.arange(0, C).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        
        # Compute the sinusoidal encodings
        pe = torch.zeros(C, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as a buffer so it is not considered a parameter but is still moved to the correct device
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        # x: input tensor with shape (batch_size, C, embed_dim)
        x = x + self.positional_encoding
        return x

class CustomTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, C, H, W):
        super(CustomTransformerEncoder, self).__init__()
        self.tokenizer = ActivationMapTokenizer(C, H, W, embed_dim)
        
        # Use Sinusoidal Positional Encoding
        self.positional_encoding = SinusoidalPositionalEncoding(C, embed_dim)
        
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layers, num_layers=num_layers
        )
        
        # Error detection head
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, 128),  # Adding a dense layer with ReLU activation
            nn.ReLU(),
            nn.Dropout(0.2),  # Adding dropout for regularization
            nn.Linear(128, 2)  # Output layer for binary classification
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Error detection output
        x = self.classification_head(x.mean(dim=1))  # Aggregate over channels
        return x

# Example usage
# model = CustomTransformerEncoder(embed_dim=512, num_heads=8, num_layers=4, C=256, H=180, W=180)
# activation_maps = torch.randn(8, 256, 180, 180)  # Example batch
# output = model(activation_maps)
# print(output.shape)  # Output shape should be (batch_size, 2)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ActivationMapTokenizer(nn.Module):
#     def __init__(self, C, H, W, embed_dim):
#         super(ActivationMapTokenizer, self).__init__()
#         self.C = C
#         self.H = H
#         self.W = W
#         self.embed_dim = embed_dim
        
#         # Linear projection of flattened tokens to embed_dim
#         self.projection = nn.Linear(H * W, embed_dim)

#     def forward(self, x):
#         # x: input activation map with shape (batch_size, C, H, W)
#         batch_size = x.size(0)
#         # print(x.shape)
#         # Flatten each channel to create tokens of shape (batch_size, C, H*W)
#         x = x.view(batch_size, self.C, -1)
#         # print(x.shape)
#         # Project flattened tokens to the desired embedding dimension
#         x = self.projection(x)  # Now shape (batch_size, C, embed_dim)
#         # print(x.shape)
#         return x

# class CustomTransformerEncoder(nn.Module):
#     def __init__(self, embed_dim, num_heads, num_layers, C, H, W):
#         super(CustomTransformerEncoder, self).__init__()
#         self.tokenizer = ActivationMapTokenizer(C, H, W, embed_dim)
#         self.positional_encoding = nn.Parameter(torch.zeros(1, C, embed_dim))
        
#         self.transformer_layers = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=num_heads
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             self.transformer_layers, num_layers=num_layers
#         )
        
#         # Error detection head
#         self.classification_head = nn.Linear(embed_dim, 2)  # for binary classification

#     def forward(self, x):
#         x = self.tokenizer(x)
#         x += self.positional_encoding
        
#         # Transformer encoding
#         x = self.transformer_encoder(x)
        
#         # Error detection output
#         x = self.classification_head(x.mean(dim=1))
#         return x

# # # Example usage
# # model = TransformerEncoder(embed_dim=512, num_heads=8, num_layers=4, C=256, H=180, W=180)
# # activation_maps = torch.randn(8, 256, 180, 180)  # Example batch
# # output = model(activation_maps)
# # print(output.shape)  # Output shape should be (batch_size, 1)
