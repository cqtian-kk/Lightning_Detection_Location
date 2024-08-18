#%%
import os
import sys
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from tools import plotvalue,creategrid,StandardDN,plotresult3Dcube,plotresult3Dslices
import random
from tqdm import tqdm

def generate_two_random_pulse_signal(num_signals,signal_size):
    # 生成随机的脉冲位置，保证均匀分布
    indices = np.linspace(0, signal_size - 1, num_signals * 2, dtype=int)
    np.random.shuffle(indices)
    indices1 = indices[:num_signals]
    indices2 = indices[num_signals:]

    # 生成信号
    signal1 = np.zeros(signal_size)
    signal1[indices1] = np.random.uniform(-1, 1, size=signal_size)[indices1]

    signal2 = np.zeros(signal_size)
    signal2[indices2] = np.random.uniform(-1, 1, size=signal_size)[indices2]

    return signal1,signal2


#data size (6,1000)
def add_noises_with_snr(data, noise, snr_db):
    # 计算信号功率
    signal_power = np.mean(data ** 2, axis = 1)
    
    # 计算噪声功率
    noise_power = np.mean(noise ** 2, axis = 1)
    
    # 根据信噪比计算噪声的标准差
    snr = 10 ** (snr_db / 10)  # 将信噪比从dB转换为线性值
    desired_noise_power = signal_power / snr
    scale_factor = np.sqrt(desired_noise_power / noise_power)
    
    # 缩放噪声并添加到数据中
    noisy_data = data + scale_factor[:, np.newaxis] * noise
    
    return noisy_data

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
def lowfnoises(rand_num1,rand_num2,A):
    # 生成随机输入点
    num_points = np.random.randint(rand_num1, rand_num2+1)  # 生成3-5个随机点
    x_points = np.sort(np.random.uniform(0, 1000, num_points))
    y_points = A*np.random.uniform(-1, 1, num_points)

    # 使用多项式拟合平滑曲线
    degree = 4  # 多项式次数
    coeffs = np.polyfit(x_points, y_points, degree)
    poly = np.poly1d(coeffs)

    # 生成平滑曲线上的点
    x_smooth = np.linspace(min(x_points), max(x_points), 1000)
    y_smooth = poly(x_smooth)
    return y_smooth
def Createnoise(noises1, noises2, noisenum, pulse_nums, A):
    noise_conv = np.zeros((noisenum,1000))
    # 初始化选取计数器和结果列表
    selection_count = 0
    selected_values = []
    array1 = np.arange(len(noises1))
    array2 = np.arange(len(noises2))
    # 进行随机选择
    while selection_count < noisenum:
        if selection_count < len(array1) and selection_count < len(array2):
            # 从两个数组中分别随机选择一个值
            value1 = np.random.choice(array1)
            value2 = np.random.choice(array2)
            selected_values.append([value1, value2])
            selection_count += 1
        else:
            # 选择次数超过数组大小时，可以重复选择
            value1 = np.random.choice(array1)
            value2 = np.random.choice(array2)
            selected_values.append([value1, value2])
            selection_count += 1
    for i in range(noisenum):
        w1 = noises1[selected_values[i][0]]
        w2 = noises2[selected_values[i][1]]
        r1,r2 = generate_two_random_pulse_signal(pulse_nums[i],901)
        trace_conv1 = np.convolve(w1, r1, mode='full')[:1000]
        trace_conv2 = np.convolve(w2, r2, mode='full')[:1000]
        noise_conv[i] = trace_conv1*A+trace_conv2*A+A*np.random.normal(0, 0.05, 1000) 
    return noise_conv
def shapenoise(noise):
    n_channels = noise.shape[0]//6
    data_noise = np.zeros((n_channels,6,1000))
    for idx in range(n_channels):
        data_noise[idx,0] = noise[idx*6]
        data_noise[idx,1] = noise[idx*6+1]
        data_noise[idx,2] = noise[idx*6+2]
        data_noise[idx,3] = noise[idx*6+3]
        data_noise[idx,4] = noise[idx*6+4]
        data_noise[idx,5] = noise[idx*6+5]
    return data_noise
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
noises_sample2 = np.loadtxt('noise/noise_NBEIBP1200_sample2.npy')
noises_sample4 = np.loadtxt('noise/noise_NBEIBP1200_sample4.npy')
noises_sample8 = np.loadtxt('noise/noise_NBEIBP1200_sample8.npy')

folder_path = './dataNBEIBP_clear'

lowfnoise_rand1 = 10
lowfnoise_rand2 = 25
pulse_nums_min = 5
pulse_nums_max = 15
random_integer_num = 50


for file_name in os.listdir(folder_path):
    if file_name.endswith(".npz"):
        file_path = os.path.join(folder_path, file_name)
        # 读取数据文件
        ligdata_c = np.load(file_path)
        ligdata_clears = ligdata_c['d1']
        label_clear = ligdata_c['d2']
        batch = ligdata_clears.shape[0]
        #随机移动波形
        ligdata_clear = np.zeros((len(ligdata_clears),6,1000))
        for i in range(len(ligdata_clears)):
            random_integer = random.randint(-1*random_integer_num, random_integer_num)
            ligdata_clear[i] = shiftfill(ligdata_clears[i], random_integer)

        #低频噪声
        data_lowfnoise = np.zeros((batch*6,1000))
        for i in tqdm(range(batch*6)):
            data_lowfnoise[i] = lowfnoises(lowfnoise_rand1,lowfnoise_rand2,6)
        pulse_nums = np.random.randint(pulse_nums_min, pulse_nums_max, batch*6)
        data_noise = Createnoise(noises_sample8,noises_sample4,batch*6,pulse_nums,10)+data_lowfnoise
        #只有底层小噪声+低频
        data_noises = shapenoise(data_noise)
        dataforlig = np.zeros((batch,6,1000))

        for m in tqdm(range(batch)):
            random_array = np.random.uniform(10, 5, size=(6,))
            noises = data_noises[m]
            # dataforlig[m] = add_noises_with_snr(ligdata_clears[m], noises, random_array)
            dataforlig[m] = add_noises_with_snr(ligdata_clear[m], noises, random_array)
        np.savez('./dataset/'+file_name[:-4]+'lowf'+str(lowfnoise_rand1)+'-'+
                 str(lowfnoise_rand2)+'_snrs'+str(pulse_nums_min)+'_'+str(pulse_nums_max)+'.npz',d1 = dataforlig, d2 = label_clear)



# %%
