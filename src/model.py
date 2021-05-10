import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Coupling layer

class Coupling_layer_const_det(nn.Module):        
    def __init__(self, input_dim, n_layers, mask_type, hidden_dim=1024):
        super(Coupling_layer_const_det, self).__init__()
        
        self.mask = self.get_mask(input_dim, mask_type)
        
        a = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1)]
        for i in range(n_layers-2):
            a.append(nn.Linear(hidden_dim, hidden_dim))
            a.append(nn.LeakyReLU(0.1))
        a.append(nn.Linear(hidden_dim, input_dim))
        self.a = nn.Sequential(*a)

    def forward(self, x):
        z = x.view(x.shape[0], -1)
        h1, h2 = z * self.mask, z * (1 - self.mask)
        
        m = self.a(h1) * (1 - self.mask)
        #h1 = h1
        h2 = h2 + m
        
        z = h1 + h2
        
        return z.view(x.shape), torch.ones(1).to(device)

    def inverse(self, z):
        x = z.view(z.shape[0], -1)
        h1, h2 = x * self.mask, x * (1 - self.mask)
        
        m = self.a(h1) * (1 - self.mask)
        #h1 = h1
        h2 = h2 - m
        
        x = h1 + h2 
        
        return x.view(z.shape)
    
    def get_mask(self, input_dim, mask_type):
        self.mask = torch.zeros(input_dim)
        if mask_type == 0:
            self.mask[::2] = 1
        
        elif mask_type == 1:
            self.mask[1::2] = 1
        
        return self.mask.view(1,-1).to(device).double()
    
    
class Coupling_layer_var_det(nn.Module):        
    def __init__(self, input_dim, n_layers, mask_type, hidden_dim=1024):
        super(Coupling_layer_var_det, self).__init__()
        self.mask = self.get_mask(input_dim, mask_type)
        
        m = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1)]
        for i in range(n_layers-2):
            m.append(nn.Linear(hidden_dim, hidden_dim))
            m.append(nn.LeakyReLU(0.1))
        m.append(nn.Linear(hidden_dim, input_dim))
        m.append(nn.Tanh())
        self.m = nn.Sequential(*m)
        
        a = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1)]
        for i in range(n_layers-2):
            a.append(nn.Linear(hidden_dim, hidden_dim))
            a.append(nn.LeakyReLU(0.1))
        a.append(nn.Linear(hidden_dim, input_dim))
        self.a = nn.Sequential(*a)
        
    def forward(self, x):
        z = x.view(x.shape[0], -1)
        h1, h2 = z * self.mask, z * (1 - self.mask)
        
        m = self.m(h1) * (1 - self.mask)
        a = self.a(h1) * (1 - self.mask)
        
        #h1 = h1
        h2 = h2 * torch.exp(m) + a
        
        z = h1 + h2
        
        logdetJ = torch.sum(m, dim=1).view(-1,1)
        
        return z.view(x.shape), logdetJ
    
    def inverse(self, z):
        x = z.view(z.shape[0], -1)
        h1, h2 = x * self.mask, x * (1 - self.mask)
        
        m = self.m(h1) * (1 - self.mask)
        a = self.a(h1) * (1 - self.mask)
        #h1 = h1
        h2 = (h2 - a) * torch.exp(-m)
        
        x = h1 + h2 
        
        return x.view(z.shape)
    
    def get_mask(self, input_dim, mask_type):
        self.mask = torch.zeros(input_dim)
        if mask_type == 0:
            self.mask[::2] = 1
        
        elif mask_type == 1:
            self.mask[1::2] = 1
        
        return self.mask.view(1,-1).to(device).double()
    

## Model
    
class FlowModel(nn.Module):
    def __init__(self, det_type, input_dim, n_layers, n_couplings, hidden_dim):
        super(FlowModel, self).__init__()
        assert det_type in ['const', 'var'], "Wrong det_type!"
        
        self.det_type = det_type
        self.input_dim = input_dim
        
        self.coupling_layers = []
        for i in range(n_couplings):
            if det_type == 'const':
                layer = Coupling_layer_const_det(input_dim, n_layers, i%2, hidden_dim).double().to(device)
            elif det_type == 'var':
                layer = Coupling_layer_var_det(input_dim, n_layers, i%2, hidden_dim).double().to(device)
            self.coupling_layers.append(layer)
        self.coupling_layers = nn.ModuleList(self.coupling_layers)
        
        
    def forward(self, x):
        return self.flow(x)
        
    
    def flow(self, x):
        logdetJ_ac = 0
        x = x.view(-1, self.input_dim).double()
        
        for layer in self.coupling_layers:
            x, logdetJ = layer(x)
            logdetJ_ac += logdetJ
            
        return x, logdetJ_ac.view(-1,1)
    
    
    def inv_flow(self, z):
        for layer in self.coupling_layers[::-1]:
            z = layer.inverse(z)
            
        return z
    
    def load_w(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
