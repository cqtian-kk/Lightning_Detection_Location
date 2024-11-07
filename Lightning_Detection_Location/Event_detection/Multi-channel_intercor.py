#%%
import os
import sys
import jpype
from jpype import *
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import torch
import time
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scipy.signal import find_peaks
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from tools import plotvalue,StandardDN
#拆分一个维度，组合其余维度
def split(arr,split):
    new_arr = np.zeros((arr.shape[0],1,arr.shape[2]))
    new_arr_res = np.zeros((arr.shape[0],arr.shape[1]-1,arr.shape[2]))
    for i in range(split.shape[0]):
        arr1 = arr[i:i+1, 0:split[i], :]
        arr2 = arr[i:i+1, split[i]:split[i]+1, :]
        arr3 = arr[i:i+1, split[i]+1:, :]
        new_arr[i] = arr2
        new_arr_res[i] = np.concatenate((arr1, arr3), axis=1)
    return new_arr,new_arr_res
#二维desplit
def desplit2(new_arr,split,new_arr_res):
    arr = np.zeros((new_arr_res.shape[0],new_arr_res.shape[1]+1))
    for i in range(split.shape[0]):
        arr[i:i+1, 0:split[i]] = new_arr_res[i:i+1, 0:split[i]]+new_arr[i:i+1]
        arr[i:i+1, split[i]:split[i]+1] = new_arr[i:i+1]
        arr[i:i+1, split[i]+1:] = new_arr_res[i:i+1, split[i]:]+new_arr[i:i+1]
    return arr
#分析选取信噪比较高的数据
def selectHighSNR(data):
    n_batch = data.shape[0]
    n_channels = data.shape[1]
    width = data.shape[2]
    
    SNR = np.zeros((n_batch,n_channels))
    width_batch = int(width/10)
    for i in range(9):
        Ei = np.sum(np.abs(data[:,:,i*width_batch:(i+1)*width_batch]) ** 2, axis=2)
        Ei1 = np.sum(np.abs(data[:,:,(i+1)*width_batch:(i+2)*width_batch]) ** 2, axis=2)
        #防止Ei=0导致计算信噪比无限大的点
        mask = Ei != 0
        SNR[mask] = SNR[mask] + Ei1[mask] / Ei[mask]
    return SNR,np.argmax(SNR,axis = 1)

def smooth(sequence, sigma):
    smoothed_sequence = gaussian_filter(sequence, sigma)
    return smoothed_sequence

def calculate_coh(s_all, sigma):
    sm1 = np.zeros(s_all.shape[1])
    sm2 = np.zeros(s_all.shape[1])
    for s in range(s_all.shape[0]):
        sm1 = sm1 + s_all[s,:]
        sm2 = sm2 + s_all[s,:]*s_all[s,:]
    sm1 /= s_all.shape[0]
    sm2 /= s_all.shape[0]
    #smoothed_sm1 = smooth(sm1, sigma)*smooth(sm1, sigma)
    smoothed_sm1 = smooth(sm1*sm1, sigma)
    smoothed_sm2 = smooth(sm2, sigma)
    coh = smoothed_sm1 / smoothed_sm2
    return coh

def check_SNRcondition(row, h1, h2):
    return np.sum(row) > h1 and np.sum(row > h2) >= row.shape[0]//2

def filter_array(arr, Erateforfilter):
    filtered_array = []
    removed_indices = []  # 存储被移除的元素索引

    prev_num = arr[0] - 201  # Initialize prev_num to a value smaller than the first element
    max_erate = 0  # 存储最大的 Erateforfilter

    for i, (num, erate) in enumerate(zip(arr, Erateforfilter)):
        if num - prev_num >= 200:
            filtered_array.append(num)
            prev_num = num
            max_erate = 0  # 重置最大的 Erateforfilter
        else:
            # 计算当前 Erateforfilter
            current_erate = num - prev_num
            if current_erate >= max_erate:
                max_erate = current_erate
                # 更新 filtered_array 中最后一个元素
                filtered_array[-1] = num
            removed_indices.append(i)

    # 生成同 arr 大小的布尔数组
    keep_mask = np.ones(len(arr), dtype=bool)
    keep_mask[removed_indices] = False

    return np.array(filtered_array), keep_mask


def selectSNRforcoh(data_Hrate, Argmaxidx, h_1, h_2):
    Erate = np.zeros((len(Argmaxidx),data_Hrate.shape[1]))
    E_all = np.sum(np.abs(data_Hrate) ** 2, axis=2)
    num = data_Hrate.shape[2]//10
    for idx in range(len(Argmaxidx)):
        #防止总能量E_all=0导致计算信噪比无限大的点
        mask = E_all[idx] != 0
        if Argmaxidx[idx] <= num//2:
            Erate[idx][mask] = np.sum(np.abs(data_Hrate[idx,:,:num]) ** 2, axis=1)[mask]/E_all[idx][mask]
        elif Argmaxidx[idx] >= data_Hrate.shape[2]-num//2:
            Erate[idx][mask] = np.sum(np.abs(data_Hrate[idx,:,-num:]) ** 2, axis=1)[mask]/E_all[idx][mask]
        else:
            Erate[idx][mask] = np.sum(np.abs(data_Hrate[idx,:, \
                Argmaxidx[idx]-num//4:Argmaxidx[idx]+3*num//4]) ** 2, axis=1)[mask]/E_all[idx][mask]
    SNRforcoh = np.apply_along_axis(check_SNRcondition, axis=1, arr=Erate, h1=h_1, h2=h_2)
        
    return SNRforcoh,Erate

def selectMidEforfilter(data_SNRforcoh):
    Erate = np.zeros((data_SNRforcoh.shape[0],data_SNRforcoh.shape[1]))
    E_all = np.sum(np.abs(data_SNRforcoh) ** 2, axis=2)
    num = data_SNRforcoh.shape[2]
    for idx in range(data_SNRforcoh.shape[0]):
        #防止总能量E_all=0导致计算信噪比无限大的点
        mask = E_all[idx] != 0
        Erate[idx][mask] = np.sum(np.abs(data_SNRforcoh[idx,:,2*num//5:3*num//5]) ** 2, axis=1)[mask]/E_all[idx][mask]
            
    return Erate

# 生成高斯窗函数
def gaussian_window(length, center):
    sigma = 250
    x = np.arange(length)
    window = np.exp(-0.5 * ((x - center) / sigma)**2)
    return window

#EMD_ 时间是从400ms开始的，采样间隔是sample
def recutdata(usedata, starttime, Argmaxidx, time_Hrate, meanshift_Hrate, winpoint, Erateforfilter):
    meanshift = np.hstack((meanshift_Hrate, np.zeros((meanshift_Hrate.shape[0], 1))))
    dtwshift = np.sort(meanshift,axis = 1)[:,2]
    startid = np.round(((time_Hrate-starttime)/sample/1e3-winpoint/2+dtwshift+Argmaxidx)).astype(int)
    startid,keep_mask = filter_array(startid, Erateforfilter)#滤除重复数据
    data = np.zeros((startid.shape[0],meanshift_Hrate.shape[1]+1,winpoint))
    time_final = startid*sample*1e3+starttime
    for idx in range(startid.shape[0]):
        data[idx] = usedata[startid[idx]:startid[idx]+winpoint,1:7].T
    return data,time_final,keep_mask
#加高斯窗
def addgausswin(data, maxh_value):
    result = np.zeros_like(data)
    shift = np.sort(maxh_value,axis = 1)[:,2]-data.shape[2]/2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            result[i,j] = data[i,j]*gaussian_window(data.shape[2],maxh_value[i,j]-shift[i])
    return result
#寻找波峰波谷点并统计滤波
def find_peak_valley_filterdata(waveform,halfwin):
    wavelen = len(waveform)
    waveform_mask = np.zeros_like(waveform)
    # 找到波峰和波谷点
    peaks, _ = find_peaks(waveform, height=None)
    valleys, _ = find_peaks(-waveform, height=None)
    if len(peaks) == 0 or len(valleys) == 0:
        return waveform_mask

    extremum = np.sort(np.concatenate((peaks,valleys)))
    # 计算波峰和波谷的振幅
    extremum_amplitudes = waveform[extremum]

    # 计算波峰和波谷振幅的平均值（μ）和标准差（σ）
    extremum_mean = np.mean(extremum_amplitudes)
    extremum_std = np.std(extremum_amplitudes)
    extremum_max = np.max(extremum_amplitudes)
    extremum_min = np.min(extremum_amplitudes)
    # 计算μ±σ的振幅范围
    extremum_threshold = (extremum_mean - extremum_std, extremum_mean + extremum_std)
    # extremum_threshold = (extremum_min*0.8, extremum_max*0.8)
    #根据halfwin和阈值进行滤波
    for i in range(len(extremum)):
        if waveform[extremum[i]]<extremum_threshold[0] or waveform[extremum[i]]>extremum_threshold[1]:
            if extremum[i]<halfwin:
                waveform_mask[0:extremum[i]+halfwin] = np.ones(extremum[i]+halfwin)
            elif extremum[i]>wavelen-halfwin:
                waveform_mask[extremum[i]-halfwin:] = np.ones(wavelen-extremum[i]+halfwin)
            else:
                waveform_mask[extremum[i]-halfwin:extremum[i]+halfwin] = np.ones(halfwin*2)
    return waveform_mask*waveform
#输入waveform（1000，）h是(0-1),越接近1 SNR越好
def find_peak_valley_createmask(waveform,h):
    maskresult = 0
    # 找到波峰和波谷点
    peaks, _ = find_peaks(waveform, height=None)
    valleys, _ = find_peaks(-waveform, height=None)
    if len(peaks) == 0 or len(valleys) == 0:
        return maskresult

    extremum = np.sort(np.concatenate((peaks,valleys)))
    # 计算波峰和波谷的振幅
    extremum_amplitudes = waveform[extremum]

    # 计算波峰和波谷振幅的平均值（μ）
    extremum_mean = np.mean(np.abs(extremum_amplitudes))
    extremum_max = np.max(np.abs(extremum_amplitudes))
    if (extremum_max-extremum_mean)/extremum_max >h:
        maskresult = 1
    return maskresult
#%%读取、筛选高信噪比数据
EMD_ = np.loadtxt('./data/EMD400-1000ns_100_500.txt')
EMD_raw = np.loadtxt('./data/EMD400-1000ns_3_500.txt')
# for i in range(EMD_.shape[1]):
#     EMD_[:,i] = StandardDN(EMD_[:,i])

sample = 0.0000002  #采样间隔s
window = 200#us
step = 20#us
winpoint = int(window/sample*1e-6)
stepoint = int(step/sample*1e-6)
EMD = EMD_[1000:]
n_channels = (EMD.shape[0]-winpoint)//stepoint

data = np.zeros((n_channels,6,winpoint))

datatime = np.zeros(n_channels)
for idx in range(n_channels):
    value = EMD[idx*stepoint:idx*stepoint+winpoint,1:].T
    datatime[idx] = EMD[idx*stepoint,0]
    data[idx] = value

SNR,SHSNR = selectHighSNR(data)
#筛选高信噪比信号，我们把所有信噪比大于100的波形段视作可能信号段
HSNRid = np.apply_along_axis(check_SNRcondition, axis=1, arr=SNR, h1=60, h2=10)
data_SNR = data[HSNRid]
datatime_SNR = datatime[HSNRid]
SHSNR_SNR = SHSNR[HSNRid]
#%%去底层噪声
# data_SNR_filter = np.zeros_like(data_SNR)
# for i in range(data_SNR.shape[0]):
#     for j in range(data_SNR.shape[1]):
#         data_SNR_filter[i,j,:] = find_peak_valley_filterdata(data_SNR[i,j,:],halfwin = 12)


reference,query = split(data_SNR,SHSNR_SNR)
print('reference.shape = ',reference.shape)

#%%-------------------------使用jtk完成dtw--------------------
#获得默认jvm路径，即jvm.dll文件路径
jvmPath = jpype.getDefaultJVMPath()
#java扩展包路径
jvmArg = "-Djava.class.path=edu-mines-jtk-1.0.0.jar"
if not jpype.isJVMStarted():
    #启动Java虚拟机
    jpype.startJVM(jvmPath,'-ea',jvmArg)
#获取相应的Java类
# smax: maximum shifts
# sr: reference trace
# st: trace to be shfited
# sx: extimated shifts
# ss: shifted version of the trace st
DynamicWarping = JClass("edu.mines.jtk.dsp.DynamicWarping")
smax = 220
dw = DynamicWarping(-smax,smax)
dw.setStrainMax(0.1)
dw.setErrorSmoothing(2)
dw.setShiftSmoothing(2)
dw.setErrorExponent(4)
result = np.zeros((reference.shape[0],query.shape[1],winpoint))
meanshift = np.zeros((reference.shape[0],query.shape[1]))
for j in tqdm(range(reference.shape[0])):
    for i in range(query.shape[1]):
        sx = dw.findShifts(reference[j,0,:],query[j,i,:]) 
        meanshift[j][i] = np.mean(sx)
        result[j,i,:] = dw.applyShifts(sx,query[j,i,:])

# shutdownJVM()
data_dtw = np.concatenate((reference, result), axis=1)
print(data_dtw.shape)

# plot
# for i in range(data_dtw.shape[0]):
#     plotvalue(data_dtw[i].T, 6)
#%%--------------------------多道互相关-------------------
sigma = 25  # Adjust the window size as needed
mcorresult = np.zeros((data_SNR.shape[0],data_SNR.shape[2]))
for i in range(data_SNR.shape[0]):
    mcorresult[i] = calculate_coh(data_dtw[i], sigma)
  
Maxre = np.max(mcorresult,axis = 1)
Meanre = np.mean(mcorresult,axis = 1)
Argmaxre = np.argmax(mcorresult,axis = 1)
rate = Maxre/Meanre
#根据比值和最大值进行第一轮筛选
Hrateid = np.logical_and(rate >1.5, Maxre > 0.40)
data_Hrate = data_dtw[Hrateid]
Argmaxidx_Hrate = Argmaxre[Hrateid]
#根据coh结果所在能量与全波形能量比进行第二轮筛选
SNRforcoh,Erate = selectSNRforcoh(data_Hrate, Argmaxidx_Hrate, 2, 0.45)
data_SNRforcoh = data_Hrate[SNRforcoh]
Erateforfilter = selectMidEforfilter(data_SNRforcoh)#去除重复数据

time_Hrate = datatime_SNR[Hrateid][SNRforcoh]
meanshift_Hrate = meanshift[Hrateid][SNRforcoh]
Argmaxidx = Argmaxre[Hrateid][SNRforcoh]
#考虑到dtw对齐的一般是波形的起始点，故加10个点
data_final,time_final,keep_mask = recutdata(EMD_, 400, Argmaxidx+10, time_Hrate, meanshift_Hrate, winpoint, np.mean(Erateforfilter,axis = 1))
data_final_raw,time_final_raw,keep_mask = recutdata(EMD_raw, 400, Argmaxidx+10, time_Hrate, meanshift_Hrate, winpoint, np.mean(Erateforfilter,axis = 1))
print('data_final = ',data_final.shape)
#制作mask
# data_final_SNR_raw,_ = selectHighSNR(data_final_raw)
# data_final_raw_Mask = (data_final_SNR_raw > 9).astype(int)
data_final_raw_Mask = np.zeros((data_final_raw.shape[0],data_final_raw.shape[1]),dtype= int)
for i in range(data_final_raw.shape[0]):
    for j in range(data_final_raw.shape[1]):
        data_final_raw_Mask[i,j] = find_peak_valley_createmask(data_final_raw[i,j,:],0.6)    
#波形加高斯脉冲
# maxh_value = desplit2(Argmaxidx,SHSNR_SNR[Hrateid][SNRforcoh],meanshift_Hrate)[keep_mask]
# result_gauss = addgausswin(data_final_raw, maxh_value)

# %%
np.savez('./location_input/EMD400-1000_3-500_271_10.npz',d1 = data_final_raw, d2 = time_final_raw, d3 = data_final_raw_Mask)
# %%
