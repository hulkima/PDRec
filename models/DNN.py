import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
#             in_dims: [94949, 1000]
#             out_dims: [1000, 94949]
#             emb_size: 10
#             time_type: 'cat'
#             norm: False
#             dropout: 0.5
        super(DNN, self).__init__()
        self.in_dims = in_dims # [49604, 1000]
        self.out_dims = out_dims # [1000, 49604]
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type # 'cat'
        self.time_emb_dim = emb_size # 10
        self.norm = norm # False

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim) # Linear(in_features=10, out_features=10, bias=True)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:] # [49614, 1000]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims # [1000, 49604]
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
#             ModuleList(
#                 (0): Linear(in_features=49614, out_features=1000, bias=True)
#             )
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
#             ModuleList(
#                 (0): Linear(in_features=1000, out_features=49604, bias=True)
#             )
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device) # torch.Size([400, 10])
        emb = self.emb_layer(time_emb) # torch.Size([400, 10])----linear project
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x) # torch.Size([400, 94949])
        h = torch.cat([x, emb], dim=-1) # torch.Size([400, 94959])
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h) # torch.Size([400, 1000])
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h # torch.Size([400, 94949])


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2 # 5
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device) # torch.Size([5])
    args = timesteps[:, None].float() * freqs[None] # torch.Size([400, 5])
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # torch.Size([400, 10])
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1) # torch.Size([400, 10])
    return embedding
