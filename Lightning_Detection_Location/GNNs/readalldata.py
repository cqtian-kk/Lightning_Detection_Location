import os
import numpy as np
def readnpz(folder_path,trainval_rate):

    # 初始化空数组用于存储数据
    train_data = np.empty((0, 6, 1000), dtype=np.float32)
    val_data = np.empty((0, 6, 1000), dtype=np.float32)

    train_label = np.empty((0, 3), dtype=np.float32)
    val_label = np.empty((0, 3), dtype=np.float32)

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npz"):
            file_path = os.path.join(folder_path, file_name)
            # 读取数据文件
            data = np.load(file_path)
            temp_data = data['d1']
            temp_label = data['d2']

            # 提取需要的部分并拼接到相应数组
            # # 随机选择800个样本生成data_train
            # indices_train_data = np.random.choice(temp_data.shape[0], size=int(temp_data.shape[0]*trainval_rate), replace=False)
            # train_data_part = temp_data[indices_train_data]
            # # 剩下的200个样本生成data_val
            # indices_val_data = np.setdiff1d(np.arange(temp_data.shape[0]), indices_train_data)
            # val_data_part = temp_data[indices_val_data]

            # train_label_part = temp_label[indices_train_data]
            # val_label_part = temp_label[indices_val_data]

            train_data_part = temp_data[0:int(temp_data.shape[0]*trainval_rate)]
            val_data_part = temp_data[int(temp_data.shape[0]*trainval_rate):]

            train_label_part = temp_label[0:int(temp_data.shape[0]*trainval_rate)]
            val_label_part = temp_label[int(temp_data.shape[0]*trainval_rate):]



            train_data = np.concatenate((train_data, train_data_part), axis=0)
            val_data = np.concatenate((val_data, val_data_part), axis=0)
            train_label = np.concatenate((train_label, train_label_part), axis=0)
            val_label = np.concatenate((val_label, val_label_part), axis=0)

    return train_data, val_data, train_label, val_label

def readrandomnpz(folder_path,trainval_rate):

    # 初始化空数组用于存储数据
    train_data = np.empty((0, 6, 1000), dtype=np.float32)
    val_data = np.empty((0, 6, 1000), dtype=np.float32)

    train_label = np.empty((0, 3), dtype=np.float32)
    val_label = np.empty((0, 3), dtype=np.float32)

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npz"):
            file_path = os.path.join(folder_path, file_name)
            # 读取数据文件
            data = np.load(file_path)
            temp_data = data['d1']
            temp_label = data['d2']

            # 提取需要的部分并拼接到相应数组
            # # 随机选择800个样本生成data_train
            indices_train_data = np.random.choice(temp_data.shape[0], size=int(temp_data.shape[0]*trainval_rate), replace=False)
            train_data_part = temp_data[indices_train_data]
            # 剩下的200个样本生成data_val
            indices_val_data = np.setdiff1d(np.arange(temp_data.shape[0]), indices_train_data)
            val_data_part = temp_data[indices_val_data]

            train_label_part = temp_label[indices_train_data]
            val_label_part = temp_label[indices_val_data]

            # train_data_part = temp_data[0:int(temp_data.shape[0]*trainval_rate)]
            # val_data_part = temp_data[int(temp_data.shape[0]*trainval_rate):]ce 

            # train_label_part = temp_label[0:int(temp_data.shape[0]*trainval_rate)]
            # val_label_part = temp_label[int(temp_data.shape[0]*trainval_rate):]



            train_data = np.concatenate((train_data, train_data_part), axis=0)
            val_data = np.concatenate((val_data, val_data_part), axis=0)
            train_label = np.concatenate((train_label, train_label_part), axis=0)
            val_label = np.concatenate((val_label, val_label_part), axis=0)

    return train_data, val_data, train_label, val_label

def mask_array(n):
    # 创建一个所有元素初始化为1
    array = np.ones((n,6),dtype=int)
    for i in range(n):
        num_zeros = np.random.choice([0, 1], p=[0.8, 0.2])  # 随机生成0-1个0
        # num_zeros = 1
        zero_indices = np.random.choice(6, num_zeros, replace=False)  # 随机选择位置
        array[i,zero_indices] = 0  # 将选中的位置设置为0
    return array