import os
import sys
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

import torch_geometric.data as gdata
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.loader as loader
from torch_geometric.data import DataLoader
import warnings

sys.path.append('../')
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
# Fixing random state for reproducibility
np.random.seed(56)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from tools import plotvalue
from loc_dataloader import *
from Trans_model_PLAN2 import *
# from Trans_model_noagg45 import *

from readalldata import *



def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 256, type=int, help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')#501
    parser.add_argument('--LR', type=float, default=0.0002, help='Learning rate.')
    parser.add_argument('--save_model_interval', type=int, default=20, help='save_model_interval.')
    parser.add_argument("--board_path", default='./loss_path/loss_demo/', help="board_path")
    parser.add_argument("--save_model_path",default='./mode_path/mode_demo/', help="Save_model_path")
    parser.add_argument("--save_model_name", default = 'model_', help="Save_model_name")
    
    args = parser.parse_args()

    return args
    
def main(args):
    
    print('save_model_path = ',args.save_model_path)
    #load data
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    # 文件夹路径
    folder_path = "../Datasets/dataset" 
    train_data, val_data, train_label, val_label = readrandomnpz(folder_path = folder_path, trainval_rate = 0.8)

    print('train_data, val_data, train_label, val_label = ',train_data.shape, val_data.shape, train_label.shape, val_label.shape)

    val_mask = mask_array(val_data.shape[0])
    # np.savez(args.save_model_path+'/temp_traindata.npz',d1 = train_data, d2 = train_label)
    # np.savez(args.save_model_path+'/temp_valdata.npz',d1 = val_data, d2 = val_label)
    np.savez(args.save_model_path+'/val_mask.npz',d1 = val_mask)

    train_dataset = MyGNNDataset(train_data, train_label)
    val_dataset = MyGNNDataset_norandmask(val_data, val_label, val_mask)

    train_loader = loader.DataLoader(train_dataset, batch_size=200,shuffle = True,num_workers = 2)
    val_loader = loader.DataLoader(val_dataset, batch_size=200,shuffle = False,num_workers = 2)
    
    trainbt = len(train_loader)
    testbt = len(val_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    
    writer = SummaryWriter(args.board_path)
    # load model
    model = baselineNet_paper_Trans().to(device)
    lr_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    print('learning rate = {}'.format(args.LR))
    scheduler = lr_scheduler.StepLR(optimizer,step_size=30,gamma = 0.9)
    epoch_total = args.epochs
    loss_total = np.zeros(epoch_total)
    loss_total_test = np.zeros(epoch_total)
    print('Finished data and model praparing , Begin Training!')
    criterion = nn.MSELoss().cuda()
    # pytorch_ssim.SSIM
    # criterion = nn.L1Loss().cuda()
    
    BEST_DIS = 200
    
    for epoch in range(epoch_total):
        
        model.train()
        err_acc_train = 0
        err_acc_val = 0
        loss_all = 0
        for mydata in train_loader:
            mydata = mydata.to(device)
            optimizer.zero_grad()
            output = model(mydata)
            loss = criterion(output, mydata.y)
            loss.backward()
            loss_all += loss.item()           
            optimizer.step()
            train_dis_x,train_dis_y,train_dis_z,train_dis_total = unscale(output,mydata.y)
            err_acc_train += train_dis_total
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        model.eval()
        loss_all_test = 0
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            loss_test = criterion(output, data.y)
            loss_all_test += loss_test.item()
            test_dis_x,test_dis_y,test_dis_z,test_dis_total = unscale(output,data.y)
            err_acc_val += test_dis_total

        scheduler.step()
        loss_total[epoch] = loss_all /trainbt
        loss_total_test[epoch] =  loss_all_test / testbt
        err_acc_train  /= trainbt
        err_acc_val /= testbt
        
        end = time.time()
        
        writer.add_scalars('loss',{'train_loss': loss_total[epoch],
                                   'test_loss': loss_total_test[epoch]}, epoch)
        
        writer.add_scalars('err',{'train': err_acc_train,
                                  'test': err_acc_val}, epoch)
        
        if epoch%1==0:
            print('Epoch: {:04d}, Loss: {:.5f} , Val_Loss: {:.5f}, train_x:{:.4f},train_y:{:.4f},train_z:{:.4f},test_x:{:.4f},test_y:{:.4f},test_z:{:.4f}'.
                  format(epoch, loss_total[epoch], loss_total_test[epoch], train_dis_x, train_dis_y, train_dis_z,test_dis_x, test_dis_y, test_dis_z))
        if epoch%(args.save_model_interval)==0:
            save_model_name = args.save_model_name + str(epoch) + '.pt'
            save_model(args.save_model_path,save_model_name,model,optimizer,epoch)
        if err_acc_val < BEST_DIS:
            BEST_DIS = err_acc_val
            save_model_name = args.save_model_name + str(epoch) + '-' + str(BEST_DIS) + '.pt'
            save_model(args.save_model_path,save_model_name,model,optimizer,epoch)  
            
    print('Finished Training!')
#     np.savez(args.loss_name,loss_total,loss_total_test)    
    print('save_model_name = ',args.save_model_name)
    writer.close()
    

def unscale(label,pred):
    MAX_X = 20.5
    MIN_X = -20
    MAX_Y = 20.5
    MIN_Y = -20
    MAX_Z = 14.5
    MIN_Z = 0
    
    label = label.cpu().detach().numpy().copy()
    pred = pred.cpu().detach().numpy().copy()  
    
    pred[:,0] = pred[:,0] *(MAX_X - MIN_X) + MIN_X
    pred[:,1] = pred[:,1] *(MAX_Y - MIN_Y) + MIN_Y
    pred[:,2] = pred[:,2] *(MAX_Z - MIN_Z) + MIN_Z
    
    label[:,0] = label[:,0] *(MAX_X - MIN_X) + MIN_X
    label[:,1] = label[:,1] *(MAX_Y - MIN_Y) + MIN_Y
    label[:,2] = label[:,2] *(MAX_Z - MIN_Z) + MIN_Z
    
    dis_x = np.abs((pred[:,0] - label[:,0])).mean()
    dis_y = np.abs((pred[:,1] - label[:,1])).mean()    
    dis_z = np.abs((pred[:,2] - label[:,2])).mean()  
    dis_total = dis_x + dis_y + dis_z
    return dis_x,dis_y,dis_z,dis_total      


                  

if __name__ == '__main__':
    args = read_args()
    main(args)

    

