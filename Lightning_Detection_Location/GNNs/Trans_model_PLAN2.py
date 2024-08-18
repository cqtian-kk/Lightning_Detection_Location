from typing import Optional, Tuple, Union
import os
import time
import math
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.datasets as gdatasets

from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable
from torch_sparse import SparseTensor, set_diag, matmul

import torch_geometric.nn as gnn 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import HeteroLinear

from torch_geometric.typing import OptPairTensor, Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax,unbatch
from torch_geometric.nn.inits import glorot, zeros


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
# Fixing random state for reproducibility
np.random.seed(19680801)

def load_model(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def save_model(path,modelname,model,optimizer,epoch):
    path = os.path.join(path,modelname)
    state = {'net': model.state_dict(),'optimizer': optimizer.state_dict(), 'epoch':epoch}
    torch.save(state, path)
    

class Conv_downsample(nn.Module):
    def __init__(self,in_ch,out_ch,ks = 3, poolstride = 2,n_layers = 2,ifpool = True):
        super(Conv_downsample, self).__init__()   
        layers = [
            nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size=ks,stride=1,padding=int(ks/2)),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(p=0.15),
        ]
        
        for i in range((n_layers - 1)):
            layers.append(nn.Conv1d(in_channels=out_ch,out_channels=out_ch,kernel_size=(ks),stride=1,padding=int(ks/2)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.15))
        
        if ifpool == True:
            layers.append(
                nn.MaxPool1d(ks, stride=poolstride,padding = int(ks/2))
            )
        
        self.multiconv=nn.Sequential(*layers) 
        

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        # x = x.unsqueeze(0)
        out = self.multiconv(x)
        return out
    


class Conv_upsample(nn.Module):
    def __init__(self, in_ch, out_ch, ks=7, upsample_scale=2, n_layers=2):
        super(Conv_upsample, self).__init__()
        layers = []
        
        for i in range(n_layers):
            layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, stride=1, padding=int(ks/2)))
            # layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.15))
            in_ch = out_ch  # Update in_ch for the next layer
        
        layers.append(nn.ConvTranspose1d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, stride=upsample_scale, padding=int(ks/2)))
        layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.15))
        
        self.multiconv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.multiconv(x)
        return out
   
class Linear_BN_relu(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Linear_BN_relu, self).__init__()   
        layers = [
            nn.Linear(in_ch,out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(p=0.15),
        ]
        self.linear=nn.Sequential(*layers) 
        
    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        # x = x.unsqueeze(0)
        out = self.linear(x)
        return out


class lstmsample(nn.Module):
    def __init__(self, in_ch, out_ch, num_layers=2):
        super(lstmsample, self).__init__()
        self.lstm = nn.LSTM(in_ch,out_ch,num_layers)

    def forward(self, x):
        out,(h,c) = self.lstm(x)
        out = out.permute(1, 0, 2)
        return out





    
class baselineNet_paper_Trans(torch.nn.Module):
    def __init__(self):
        super(baselineNet_paper_Trans, self).__init__()
        
        
        self.conv1 = Conv_downsample(1,4,ks = 3, poolstride = 2,n_layers = 1,ifpool = True)#(Nb, Ns, 2048, 3) -> (Nb, Ns, 512, 8) -> (Nb, Ns, 128, 16) -> (Nb, Ns, 32, 32)

        self.conv4 = Conv_downsample(4,1,ks = 3, poolstride = 1,n_layers = 1,ifpool = False)#3072 768

        self.lstm = lstmsample(500,1000,2)
        self.Graph_agg1 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = 256, 
                   out_channels = 128,        
                   heads = 2,dropout = 0.1,root_weight = True,concat = False), 'x, edge_index -> x'),
        ])
           
        self.Graph_agg3 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = 128, 
                   out_channels = 64,        
                   heads = 2,dropout = 0.1,root_weight = True,concat = False), 'x, edge_index -> x'),
        ])
        self.Graph_agg4 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = 64, 
                   out_channels = 64,        
                   heads = 2,dropout = 0.1,root_weight = True,concat = False), 'x, edge_index -> x'),
        ])
        self.Graph_agg5 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = 64, 
                   out_channels = 64,        
                   heads = 2,dropout = 0.1,root_weight = True,concat = False), 'x, edge_index -> x'),
        ])
 


        self.staion_mlp = Mlp(3,128,250)
        self.total_mlp = Mlp(1250,512,256)
        
        self.mlp1 = Mlp(64,16,3)
        self.act1 = torch.nn.Sigmoid()


    def forward(self, data):
        x, edge_index, edge_attr, batch, station_pos = data.x, data.edge_index, data.edge_attr, data.batch, data.sta_pos   
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv4(x)
        x = self.lstm(x).squeeze()
        station_loc = self.staion_mlp(station_pos)
        x = torch.concat([x,station_loc],dim = -1)
        x = self.total_mlp(x)

        x = self.Graph_agg1(x.squeeze(),edge_index)

        x1 = self.Graph_agg3(x.squeeze(),edge_index)
        x2 = self.Graph_agg4(x1.squeeze(),edge_index)
        x3 = self.Graph_agg5(x2.squeeze(),edge_index)
        

        x = gmp(x1+x2+x3, batch)
        x = self.mlp1(x)
        x = self.act1(x)
        return x
    
