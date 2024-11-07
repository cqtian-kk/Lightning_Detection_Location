#%%
import os
import sys
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

#叠加生成复杂无噪数据
# 给定移动的步数 shift（正数表示向前移动，负数表示向后移动）
def shiftfill(ligdatas, step):
    # 对数组进行操作
    result = np.empty_like(ligdatas)
    for i in range(ligdatas.shape[0]):
        if step > 0:  # 向右移动
            # 保存每行的第一个元素
            fill_value = ligdatas[i, 0]
            # 将数组向右移动
            result[i, :] = np.roll(ligdatas[i, :], step)
            # 用第一个元素填充前面的空白位置
            result[i, :step] = fill_value
        else:  # 向左移动
            # 保存每行的最后一个元素
            fill_value = ligdatas[i, -1]
            # 将数组向左移动
            result[i, :] = np.roll(ligdatas[i, :], step)
            # 用最后一个元素填充后面的空白位置
            result[i, step:] = fill_value
    return result
def shiftandover_datas(shift,data,A): 
    # 向前移动数据，使用np.roll函数
    shifted_data = shiftfill(data, shift)
    shifted_data = shifted_data*A+data
    return shifted_data

#生成单道噪声
def generate_random_pulse_signal(input_num, pulse_num):
    """
    生成随机脉冲信号。
    """

    # 创建一个与输入数组大小相同的零数组
    pulse_signal = np.zeros(input_num)

    # 随机选择 num 个位置来生成脉冲
    pulse_positions = np.random.choice(input_num, pulse_num, replace=False)

    # 在选择的位置上设置脉冲为1
    pulse_signal[pulse_positions] = 1

    return pulse_signal
def Createbignoise(noises, noisenum):
    noise_conv = np.zeros((noisenum,1000))
    # 初始化选取计数器和结果列表
    selection_count = 0
    selected_values = []
    array = np.arange(len(noises))
    # 进行随机选择
    while selection_count < noisenum:
        value = np.random.choice(array)
        selected_values.append(value)
        selection_count += 1

    for i in range(noisenum):
        w = noises[selected_values[i]]
        r = generate_random_pulse_signal(601,1)
        trace_conv = np.convolve(w, r, mode='full')[:1000]
        noise_conv[i] = trace_conv
    return noise_conv
def reshape_and_randomize(arr, new_shape):
    n = new_shape[0]
    reshaped_arr = np.zeros(new_shape, dtype=arr.dtype)
    
    for i in range(n):
        random_idx = np.random.randint(0, new_shape[1])
        reshaped_arr[i, random_idx, :] = arr[i, :]
    
    return reshaped_arr
def Scalenoiseandadd(data,noises):
    selectnum = np.argmax(np.sum(np.abs(noises),axis = 1))
    A = np.max(np.abs(data[selectnum]))/np.max(np.abs(noises[selectnum]))*0.8
    newdata = data + A*noises
    return newdata

noises_sample2 = np.loadtxt('noise/noise_NBEIBP1200_sample2.npy')
noise_sample2 = Createbignoise(noises_sample2, 20)
data_noises = reshape_and_randomize(noise_sample2,(20,6,1000))

# ligdata_c is Type 1
ligdata_c = np.load('./data_clear/data0num2.npz')
ligdata_clear = ligdata_c['d1']
label_clear = ligdata_c['d2']
data2 = np.zeros_like(ligdata_clear)
data3 = np.zeros_like(ligdata_clear)
data4 = np.zeros_like(ligdata_clear)
data5 = np.zeros_like(ligdata_clear)
# ligdata_c is Type 1
np.savez('./dataset_clear/data0num2_Type1.npz',d1 = ligdata_clear, d2 = label_clear)

# Changewaveform to Type 2
for i in range(2):
    data2[i] = Scalenoiseandadd(ligdata_clear[i],data_noises[i]) 
np.savez('./dataset_clear/data0num2_Type2',d1 = data2, d2 = label_clear)

# Changewaveform to Type 3
random_integers = [random.randint(-25, -15) if random.random() < 0.5 else random.randint(15, 25) for _ in range(len(ligdata_clear))]
for i in range(len(ligdata_clear)):
    data3[i] = shiftandover_datas(random_integers[i],ligdata_clear[i],np.random.uniform(0,1))
np.savez('./dataset_clear/data0num2_Type3.npz',d1 = data3, d2 = label_clear)

# Changewaveform to Type 4
random_integers = [random.randint(-100, -50) if random.random() < 0.5 else random.randint(50, 100) for _ in range(len(ligdata_clear))]
for i in range(len(ligdata_clear)):
    data4[i] = shiftandover_datas(random_integers[i],ligdata_clear[i],np.random.uniform(0,1))
np.savez('./dataset_clear/data0num2_Type4.npz',d1 = data4, d2 = label_clear)

# Changewaveform to Type 5
random_integers = [random.randint(-500, -480) if random.random() < 0.5 else random.randint(480, 500) for _ in range(len(ligdata_clear))]
for i in range(len(ligdata_clear)):
    data5[i] = shiftandover_datas(random_integers[i],ligdata_clear[i],np.random.uniform(0,1))
np.savez('./dataset_clear/data0num2_Type5.npz',d1 = data5, d2 = label_clear)



# %%
