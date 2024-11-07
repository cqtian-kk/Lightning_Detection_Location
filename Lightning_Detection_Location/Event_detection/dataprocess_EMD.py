#%%
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD,EEMD
import pandas as pd
import numpy.fft as fft

#%%
#读取波形数据
data1 = np.loadtxt('./data/S1.txt',skiprows=1)
time1 = data1[:, 0]
value1 = data1[:, 1]
data2 = np.loadtxt('./data/S2.txt',skiprows=1)
time2 = data2[:, 0]
value2 = data2[:, 1]
data3 = np.loadtxt('./data/S3.txt',skiprows=1)
time3 = data3[:, 0]
value3 = data3[:, 1]
data4 = np.loadtxt('./data/S4.txt',skiprows=1)
time4 = data4[:, 0]
value4 = data4[:, 1]
data5 = np.loadtxt('./data/S5.txt',skiprows=1)
time5 = data5[:, 0]
value5 = data5[:, 1]
data6 = np.loadtxt('./data/S6.txt',skiprows=1)
time6 = data6[:, 0]
value6 = data6[:, 1]
print('data')

# Perform EMD decomposition
emd = EEMD()
imfs1 = emd(value1)
print(1)
imfs2 = emd(value2)
print(1)
imfs3 = emd(value3)
print(1)
imfs4 = emd(value4)
print(1)
imfs5 = emd(value5)
print(1)
imfs6 = emd(value6)

np.savetxt('./data/S1_imfs.txt', imfs1.T)
np.savetxt('./data/S2_imfs.txt', imfs2.T)
np.savetxt('./data/S3_imfs.txt', imfs3.T)
np.savetxt('./data/S4_imfs.txt', imfs4.T)
np.savetxt('./data/S5_imfs.txt', imfs5.T)
np.savetxt('./data/S6_imfs.txt', imfs6.T)
#%% 对每个台站波形进行EMD处理
import numpy as np
import pandas as pd
import numpy.fft as fft
import matplotlib.pyplot as plt
imfs = np.loadtxt('./data/S1_imfs.txt')
data_ = np.loadtxt('./data/S1.txt',skiprows=1)
time = data_[:, 0]

main_freq =  np.zeros(imfs.shape[1])
signal_fliter =  np.zeros(imfs.shape[0])
for i in range(imfs.shape[1]):
    signal = imfs[:,i]
    signal_length = len(signal)
    # 计算离散傅里叶变换
    fft_signal = fft.fft(signal)
    
    # 计算频率谱
    freq = np.fft.fftfreq(signal_length, d=(time[1]-time[0])*1e-3)
    amplitude_spectrum = np.abs(fft_signal)
    
    # 确定主频
    max_freq_index = np.argmax(amplitude_spectrum)
    main_frequency = freq[max_freq_index]
    main_freq[i] = main_frequency
    if abs(main_frequency)>=3000 and abs(main_frequency)<=500000:
        signal_fliter  = signal_fliter + signal
plt.figure(figsize=(100, 36))

plt.plot(time, signal_fliter)
data = np.vstack((time, signal_fliter))
np.savetxt('./data/S1_EMD_3_500.txt', data.T)
#%%
import numpy as np

# 读取数据
data1 = np.loadtxt('./data/S1_EMD_3_500.txt', skiprows=1)
time1 = data1[:, 0]
value1 = data1[:, 1]

data2 = np.loadtxt('./data/S2_EMD_3_500.txt', skiprows=1)
time2 = data2[:, 0]
value2 = data2[:, 1]

data3 = np.loadtxt('./data/S3_EMD_3_500.txt', skiprows=1)
time3 = data3[:, 0]
value3 = data3[:, 1]

data4 = np.loadtxt('./data/S4_EMD_3_500.txt', skiprows=1)
time4 = data4[:, 0]
value4 = data4[:, 1]

data5 = np.loadtxt('./data/S5_EMD_3_500.txt', skiprows=1)
time5 = data5[:, 0]
value5 = data5[:, 1]

data6 = np.loadtxt('./data/S6_EMD_3_500.txt', skiprows=1)
time6 = data6[:, 0]
value6 = data6[:, 1]

# 获取所有时间戳的并集
tmin = 400
tmax = np.max([time1[-1],time2[-1],time3[-1],time4[-1],time5[-1],time6[-1]])
all_times = np.linspace(tmin, tmax,int((tmax-tmin)/0.0002))

# 将每个数据集的值插入到所有时间戳中
values = np.zeros((6, len(all_times)))
values[0] = np.interp(all_times, time1, value1, left=0, right=0)
values[1] = np.interp(all_times, time2, value2, left=0, right=0)
values[2] = np.interp(all_times, time3, value3, left=0, right=0)
values[3] = np.interp(all_times, time4, value4, left=0, right=0)
values[4] = np.interp(all_times, time5, value5, left=0, right=0)
values[5] = np.interp(all_times, time6, value6, left=0, right=0)

data = np.vstack((all_times,values))
np.savetxt('./data/EMD400-1000ns_3_500.txt', data.T)