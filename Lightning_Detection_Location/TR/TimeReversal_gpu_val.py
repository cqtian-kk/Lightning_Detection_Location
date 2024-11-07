#%%
from numba import cuda
from numba import jit
import numba
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import numpy.fft as fft
import os
import sys
import timeit
from tqdm import tqdm
start = timeit.default_timer()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from computeZn import computingZn_GPU as computingZn
from computeZn import computingZns_GPU as computingZns
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from tools import creategrid
# 光速
c = 300000  # km/s
    
sample = 0.0000002  #采样间隔s
#各台站位置
sta_x = [-3.324113012, 14.97894469, 1.252397509, -11.96347912, -5.929079939, 4.98532987]
sta_y = [-0.029319449, 4.206803539, 11.66024498,4.639923346, -13.80897884, -6.668673572]
sta_z = [0.015169, 0.020862, 0.030253, 0.033826, 0.037787244, 0.033646]
stationlist = []
for i in range(len(sta_x)):
    stationlist.append([sta_x[i], sta_y[i], sta_z[i]])
stationlist = np.array(stationlist)




latgrid = 1
longrid = 1
highgrid =1
studyarea = [-20, -20, 41, 41]  # km
studydepth = [0, 15]  # km
lats = np.arange(studyarea[0],  studyarea[0]+studyarea[2], latgrid)
lons = np.arange(studyarea[1],  studyarea[1]+studyarea[3], longrid)
highs = np.arange(studydepth[0], studydepth[1], highgrid)
studygrids = creategrid(lats, lons, highs)

# #生成走时表
stationlist_repeat = np.tile(stationlist, (studygrids.shape[0],1))
studygrids_repeat = np.repeat(studygrids, stationlist.shape[0], axis = 0)
traveldis = np.sum((studygrids_repeat-stationlist_repeat)**2,axis = 1)
traveltime = np.sqrt(traveldis)/c
traveltime = traveltime.reshape(studygrids.shape[0],stationlist.shape[0])
# np.savetxt('./traveltime/traveltime{}_{}.npy'.format(studygrids.shape[0],stationlist.shape[0]), traveltime)
# #读取走时表
# traveltime = np.loadtxt('./traveltime/traveltime{}_{}.npy'.format(studygrids.shape[0],stationlist.shape[0]))

window = 200#us
step = 100#us
winpoint = int(window/sample*1e-6)
stepoint = int(step/sample*1e-6)


waveform_data = np.load('../Event_detection/location_input/EMD400-1000_3-500_271_10.npz')

waveform = waveform_data['d1']
timea = waveform_data['d2']

result = []

cuda_studygrids = cuda.to_device(studygrids)
cuda_traveltime = cuda.to_device(traveltime)
cuda_stationlist = cuda.to_device(stationlist)

for idx in tqdm(range(len(waveform))):
    value = waveform[idx]
    #--------------计算TR.GPU能量时间域------------
    Zn = np.zeros((winpoint,studygrids.shape[0]))
    cuda_Zn = cuda.to_device(Zn)
    cuda_u_list = cuda.to_device(value)
    computingZn[studygrids.shape[0],value.shape[0]](cuda_Zn,winpoint, cuda_studygrids, cuda_u_list, cuda_stationlist, sample, cuda_traveltime)
    cuda.synchronize()
    Zn = cuda_Zn.copy_to_host()   
    # 能量E
    E = np.zeros(studygrids.shape[0])
    for lm in range(studygrids.shape[0]):
        E[lm] = np.linalg.norm(Zn[:, lm])**2
    #进行二级网格搜索 
    event_index = np.argmax(E)
    event = studygrids[event_index]
    latssub = np.arange(event[0]-1,  event[0]+1, 0.05)
    lonssub = np.arange(event[1]-1,  event[1]+1, 0.05)
    highssub = np.arange(event[2]-1,  event[2]+1, 0.05)
    studygridssub = creategrid(latssub, lonssub, highssub)
    #---------------------在GPU上生成走时表---------------------------------------
    #--------------计算能量时间域
    Znsub = np.zeros((winpoint, studygridssub.shape[0]))
    cuda_Znsub = cuda.to_device(Znsub)
    cuda_studygridssub = cuda.to_device(studygridssub)
    computingZns[studygridssub.shape[0],value.shape[0]](cuda_Znsub, winpoint, cuda_studygridssub, cuda_u_list, cuda_stationlist, sample)
    cuda.synchronize()
    Znsub = cuda_Znsub.copy_to_host() 
    Esub = np.zeros(studygridssub.shape[0])
    # #释放显存
    del cuda_studygridssub
    del cuda_Znsub
    del cuda_Zn
    del cuda_u_list
    for lm in range(studygridssub.shape[0]):
        Esub[lm] = np.linalg.norm(Znsub[:, lm])**2
    event_indexsub = np.argmax(Esub)
    eventsub = studygridssub[event_indexsub]
    result.append(eventsub)
np.savez('./Location_result/TR271_10.npz',d1 = timea, d2 = np.array(result))

# %%
