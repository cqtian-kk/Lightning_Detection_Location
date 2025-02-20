#%%
import os
import torch
import numpy as np
import torch_geometric.data as gdata
import torch_geometric.loader as loader

from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
#%%
def cal_edge(pos_init):
    pos = np.zeros([pos_init**2,2])
    for i in range (pos_init**2):
        pos[i,0] = int(i/pos_init)
        pos[i,1] = int(i%pos_init)
    edge_index = torch.tensor(pos, dtype=torch.long)
    return edge_index.T
#无自循环
def cal_edge_(pos_init):
    edge_index = []

    # 生成有向边（每对节点都要有两个方向）
    for i in range(pos_init):
        for j in range(pos_init):
            if i != j:  # 不允许节点与自身相连
                edge_index.append([i, j])  # 从i到j
    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index

def distance(pos1, pos2):
    # 计算两点间的欧几里得距离
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def cal_edge_attr(sta_pos, edge_index):
    # 创建一个空的距离数组
    dis = np.zeros(edge_index.shape[1])
    
    # 计算每条边的距离
    for i in range(edge_index.shape[1]):
        # 获取当前边的起始节点和目标节点
        dis[i] = distance(sta_pos[edge_index[0, i]], sta_pos[edge_index[1, i]])

    # 归一化距离（min-max 归一化到 0 到 1 范围内）
    dis = dis / np.max(dis)  # 归一化到 0-1 范围内

    # 将距离转换为 torch 张量并返回
    return torch.tensor(dis, dtype=torch.float).view(-1, 1)




def createmask():
    # 创建一个所有元素初始化为1
    array = np.ones(6,dtype=int)
    num_zeros = np.random.choice([0, 1], p=[0.8, 0.2])  # 随机生成0-1个0
    # num_zeros = 1
    zero_indices = np.random.choice(6, num_zeros, replace=False)  # 随机选择位置
    array[zero_indices] = 0  # 将选中的位置设置为0
    return array

class MyGNNDataset(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label
        self.MAX_X = 20.5
        self.MIN_X = -20
        self.MAX_Y = 20.5
        self.MIN_Y = -20
        self.MAX_Z = 14.5
        self.MIN_Z = 0
        
        
    def __getitem__(self, index):
        
        data = self.data[index]
        label = self.label[index]
        mask = createmask()


        data = self.x_normalization(data)[mask ==1]
        label = self.y_normalization(label)
        sta_pos = self.define_position()[mask ==1]
        edge_index = cal_edge_(np.sum(mask))
        edge_attr = cal_edge_attr(sta_pos, edge_index)
        data = torch.tensor(data,dtype = torch.float)
        label = torch.tensor(label,dtype = torch.float).unsqueeze(0)
        sta_pos = torch.tensor(sta_pos,dtype = torch.float)
        
        

        return gdata.Data(x = data, y = label, sta_pos = sta_pos, 
                          edge_index = edge_index, edge_attr = edge_attr) 

    def __len__(self):
        
        return (len(self.data))





    # 归一化data
    def x_normalization(self,x,norm_channel = True):
        norm_data = np.ones_like(x)
        ### if norm each channel
        if norm_channel == True:
            for i in range(x.shape[0]):
                norm_data[i,:] = (x[i,:] - x[i,:].mean())/x[i,:].std()
            
        elif norm_channel == False:
        ### norm station
            x_mean = x.mean()
            x_std = x.std()
            norm_data = (x-x_mean)/x_std
            
        return norm_data
    
    # 归一化label
    def y_normalization(self,y):
        y[0] = (y[0] - self.MIN_X)/(self.MAX_X - self.MIN_X)
        y[1] = (y[1] - self.MIN_Y)/(self.MAX_Y - self.MIN_Y)
        y[2] = (y[2] - self.MIN_Z)/(self.MAX_Z - self.MIN_Z)
        return y
    
    # 归一化台站位置
    def define_position(self):
        sta_x = np.array([-3.324113012, 14.97894469, 1.252397509, -11.96347912, -5.929079939, 4.98532987])
        sta_y = np.array([-0.029319449, 4.206803539, 11.66024498,4.639923346, -13.80897884, -6.668673572])
        sta_z = np.array([0.015169, 0.020862, 0.030253, 0.033826, 0.037787244, 0.033646])
        sta_pos = np.concatenate([sta_x,sta_y,sta_z]).reshape(3,-1).T
        
        sta_pos[:,0] = (sta_pos[:,0] - self.MIN_X)/(self.MAX_X - self.MIN_X)
        sta_pos[:,1] = (sta_pos[:,1] - self.MIN_Y)/(self.MAX_Y - self.MIN_Y)
        sta_pos[:,2] = (sta_pos[:,2] - self.MIN_Z)/(self.MAX_Z - self.MIN_Z)
        
        return sta_pos



class MyGNNDataset_norandmask(Dataset):
    def __init__(self,data,label,mask):
        self.data = data
        self.label = label
        self.mask = mask
        self.MAX_X = 20.5
        self.MIN_X = -20
        self.MAX_Y = 20.5
        self.MIN_Y = -20
        self.MAX_Z = 14.5
        self.MIN_Z = 0
        
        
    def __getitem__(self, index):
        
        data = self.data[index]
        label = self.label[index]
        mask = self.mask[index]


        data = self.x_normalization(data)[mask ==1]
        label = self.y_normalization(label)
        sta_pos = self.define_position()[mask ==1]
        edge_index = cal_edge_(np.sum(mask))
        edge_attr = cal_edge_attr(sta_pos, edge_index)
        data = torch.tensor(data,dtype = torch.float)
        label = torch.tensor(label,dtype = torch.float).unsqueeze(0)
        sta_pos = torch.tensor(sta_pos,dtype = torch.float)
        return gdata.Data(x = data, y = label, sta_pos = sta_pos, 
                          edge_index = edge_index, edge_attr = edge_attr) 


    def __len__(self):
        
        return (len(self.data))

    # 归一化data
    def x_normalization(self,x,norm_channel = True):
        norm_data = np.ones_like(x)
        ### if norm each channel
        if norm_channel == True:
            for i in range(x.shape[0]):
                norm_data[i,:] = (x[i,:] - x[i,:].mean())/x[i,:].std()
            
        elif norm_channel == False:
        ### norm station
            x_mean = x.mean()
            x_std = x.std()
            norm_data = (x-x_mean)/x_std
            
        return norm_data
    
    # 归一化label
    def y_normalization(self,y):
        y[0] = (y[0] - self.MIN_X)/(self.MAX_X - self.MIN_X)
        y[1] = (y[1] - self.MIN_Y)/(self.MAX_Y - self.MIN_Y)
        y[2] = (y[2] - self.MIN_Z)/(self.MAX_Z - self.MIN_Z)
        return y
    
    # 归一化台站位置
    def define_position(self):
        sta_x = np.array([-3.324113012, 14.97894469, 1.252397509, -11.96347912, -5.929079939, 4.98532987])
        sta_y = np.array([-0.029319449, 4.206803539, 11.66024498,4.639923346, -13.80897884, -6.668673572])
        sta_z = np.array([0.015169, 0.020862, 0.030253, 0.033826, 0.037787244, 0.033646])
        sta_pos = np.concatenate([sta_x,sta_y,sta_z]).reshape(3,-1).T
        
        sta_pos[:,0] = (sta_pos[:,0] - self.MIN_X)/(self.MAX_X - self.MIN_X)
        sta_pos[:,1] = (sta_pos[:,1] - self.MIN_Y)/(self.MAX_Y - self.MIN_Y)
        sta_pos[:,2] = (sta_pos[:,2] - self.MIN_Z)/(self.MAX_Z - self.MIN_Z)
        
        return sta_pos