#%%
import os
import sys
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import argparse
import time
import random
import torch_geometric
from tqdm import tqdm
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import torch_geometric.data as gdatavs
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.loader as loader
from torch_geometric.data import DataLoader
import warnings
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from tools import plotvalue
sys.path.append('../')
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')


from loc_dataloader import *
from Trans_model_PLAN2 import *
from readalldata import *
from train_gnn_PLAN2 import unscale
def unscale(pred):
    MAX_X = 20.5
    MIN_X = -20
    MAX_Y = 20.5
    MIN_Y = -20
    MAX_Z = 14.5
    MIN_Z = 0
    
    pred = pred.cpu().detach().numpy().copy()  
    
    pred[:,0] = pred[:,0] *(MAX_X - MIN_X) + MIN_X
    pred[:,1] = pred[:,1] *(MAX_Y - MIN_Y) + MIN_Y
    pred[:,2] = pred[:,2] *(MAX_Z - MIN_Z) + MIN_Z
    

    
    return pred


# 文件夹路径
file_path = "../Event_detection/location_input/EMD400-1000_3-500_271_10.npz"

model_path = './mode_path/mode_PLAN2lstm_1000dconv14/model_500-1.0628212714806582.pt'

data = np.load(file_path)
temp_data = data['d1']
temp_datatime = data['d2']
temp_mask = data['d3']
# temp_mask = np.ones((temp_data.shape[0],6), dtype=int)
temp_label = np.zeros((temp_data.shape[0],3))




pre_dataset = MyGNNDataset_norandmask(temp_data, temp_label, temp_mask)

pre_loader = loader.DataLoader(pre_dataset, batch_size=200,shuffle = False,num_workers = 2)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

model = baselineNet_paper_Trans().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
model, optimizer, epoch = load_model(model_path, model, optimizer)

model.eval()
result = np.empty((0, 3), dtype=np.float32)
for data in tqdm(pre_loader):
    data = data.to(device)
    output = model(data)
    result = np.concatenate((result, unscale(output)), axis=0)
np.savez('./gnnresult/gnnresult_271_10.npz',d1 = result, d2 = temp_datatime)


# %%
