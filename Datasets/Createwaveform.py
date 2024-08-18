#%%
import os
import sys
from tqdm import tqdm
import math
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import random
from Rgridsmove import random_grids
from scipy.integrate import quad
from scipy.integrate import dblquad
import sympy as sp
import itertools

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from tools import creategrid
def Dowave(time, H1, h0, sour_z, D, t1, t2, aind, v_dlb ):

    def Electric(t):
        a = aind/t2
        if t <= t1:
            value = A*math.exp(-(a*(t-t1))**2)
        else:
            value = A*math.exp(-(a*(t-t1)*t1/(t2-t1))**2)
        return value

    def Electric_sp(t):
        a = aind/t2
        return sp.Piecewise((A * sp.exp(-(a * (t - t1))**2), t <= t1),
                                (A * sp.exp(-(a * (t - t1) * t1 / (t2 - t1))**2), True))
    def I(sour_z,t):
        return (H2-sour_z)/(H2-H1)*Electric(t-(sour_z-H1)/v_dlb)
    def I_sp(sour_z,t):
        return (H2-sour_z)/(H2-H1)*Electric_sp(t-(sour_z-H1)/v_dlb)


    def E1_inner(sour_z,t_):
        return 1/2/math.pi/e0*(2-3*math.sin(θ)**2)/R**3*I(sour_z,t_-R/c)
    def E1(time,H1,H2): 
        result = []
        for t_value in time:
            temp, error = dblquad(E1_inner, 0, t_value, H1, H2) 
            result.append(temp)
        return np.array(result)

    def E2_inner(sour_z,t):
        return 1/2/math.pi/e0*(2-3*math.sin(θ)**2)/c/R**2*I(sour_z,t-R/c)

    def E2(time,H1,H2): 
        result = []
        for t_value in time:
            temp, error = quad(E2_inner, H1, H2, args=(t_value))
            result.append(temp)
        return np.array(result)   

    def E3_inner(t_value):
        sour_z, time = sp.symbols('sour_z time')
        I_dt = sp.diff(I_sp(sour_z, time), time).subs(time, t_value-R/c)
        E3_in = 1/2/math.pi/e0*math.sin(θ)**2/c**2/R*I_dt
        return sp.lambdify(sour_z, E3_in)


    def E3(time,H1,H2): 
        result = []
        for t_value in time:
            E3_in = E3_inner(t_value)
            temp, error = quad(E3_in, H1, H2)
            result.append(temp)
        return np.array(result) 
    # 电流函数参数
    # t1 = 5e-6 # 上升常数
    # t2 = 20e-6 # 下降常数
    # aind = 19
    A = 35e3
    H2 = H1+h0
    R = math.sqrt(sour_z*sour_z+D*D)
    # 计算 cot^-1 函数的值
    θ = np.arctan(1 / np.tan(-sour_z/D))
    # v_dlb = 0.8e8 # 电流波的传播速度m/s
    e0 = 8.854187817e-12 # 真空介电常数
    c = 3e8 # m/s 光速

    e_1 = E1(time,H1,H2)
    e_2 = E2(time,H1,H2)
    e_3 = E3(time, H1, H2)
    data = e_1+e_2-e_3
    return data



def calculate_distances(source_points, stations):
    # 将源点数组扩展为（n，1，3），以便与台站数组进行广播计算
    expanded_source_points = source_points[:, np.newaxis, :]
    
    # 计算距离数组，每个元素代表一个源点到一个台站的距离
    distances = np.linalg.norm(expanded_source_points - stations, axis=2)
    
    return distances

# 光速
c = 300000 # km/s
# 闪电信号参数
t_len = 0.0002 #秒
sample = 0.0000002 # 采样间隔
pointsnum = int(t_len/sample)
time = np.arange(0, pointsnum*sample, sample)


# 台站位置坐标
sta_x = [-3.324113012, 14.97894469, 1.252397509, -11.96347912, -5.929079939, 4.98532987]
sta_y = [-0.029319449, 4.206803539, 11.66024498,4.639923346, -13.80897884, -6.668673572]
sta_z = [0.015169, 0.020862, 0.030253, 0.033826, 0.037787244, 0.033646]
stationlist = []
for i in range(len(sta_x)):
    stationlist.append([sta_x[i], sta_y[i], sta_z[i]])
stationlist = np.array(stationlist)
stanum = len(stationlist)
# 正演网格点
latgrid = 0.5
longrid = 0.5
highgrid = 0.5
studyarea = [-14, -14, 29, 29]  # km
studydepth = [6, 13]  # km
lats = np.arange(studyarea[0],  studyarea[0]+studyarea[2], latgrid)
lons = np.arange(studyarea[1],  studyarea[1]+studyarea[3], longrid)
highs = np.arange(studydepth[0], studydepth[1], highgrid)
studygrids = creategrid(lats, lons, highs)





batch = 10
liglabel = np.zeros((batch, 3))
labelforlig = np.zeros((batch*2, 3))
dataforlig = np.zeros((batch*2, 6, pointsnum))
data_clear = np.zeros((batch, 6, pointsnum))


#---------------------模拟闪电震源点-------------------
sources = np.zeros((batch,3))
# lignum代表一个窗口内模拟闪电源数量，lignum大于1时，
# random_grids能模拟多个空间位置比较近的源，这里lignum = 1，
# random_grids等效于随机选取空间中任意一点
lignum = 1
for i in range(batch):
    selected_grids = random_grids(lats, lons, highs, lignum)
    sources[i] = selected_grids[0]
# ------------------计算台站到激发点的2D距离------------
Distance = calculate_distances(sources[:,0:2],stationlist[:,0:2])
# ------------------生成所有可能的参数组合--------------
t1_range = np.arange(4.8e-6,25e-6,2e-6)  # 设置合适的范围
t1_t2_diff_range = np.arange(15e-6,37e-6,2e-6)  # t2 - t1 的范围
aind_range = np.arange(6.9,22.7,3)
v_dlb_range = np.arange(0.4e8,3e8,0.4e8) 
h0_range = np.arange(0.108,1.5,0.1)
combinations = list(itertools.product(t1_range, t1_t2_diff_range, aind_range, v_dlb_range, h0_range))
# 将组合转换为6维数组，其中第一维表示组合的总数
combinations_array = np.array(combinations)
#%% 生成闪电数据
for m in tqdm(range(batch)):
    t1 = combinations_array[m,0]
    t2 = combinations_array[m,1]+t1
    aind = combinations_array[m,2]
    v_dlb = combinations_array[m,3]
    h0 = combinations_array[m,4]
    # 生成闪电信号波形
    ligdatas = np.zeros((stanum, pointsnum))
    sour_z = sources[m][2]
    H1 = sour_z
    for i in range(stanum):
        signal = Dowave(time, H1*1000, h0*1000, sour_z*1000, Distance[m,i]*1000, t1, t2, aind, v_dlb)
        ligdatas[i] = signal 
        # ligdatas[i] = -signal 
    # 调整生成的闪电波形尽量居中
    argminidx = np.sort(np.argmin(ligdatas,axis = 1))[2]
    padded_array = np.pad(ligdatas, ((0, 0), (500, 500)), mode='constant')
    ligdatas_shift = padded_array[:,argminidx:argminidx+1000]
    liglabel[m] = sources[m]
    data_clear[m] = ligdatas_shift
    # if m % 200 ==0:
    #     np.savez('data_clear/data0zhengnum10_'+str(m)+'.npz',d1 = data_clear, d2 = liglabel)

labelforlig[0:batch] = liglabel
labelforlig[batch:] = liglabel
dataforlig[0:batch] = data_clear
dataforlig[batch:] = -data_clear
np.savez('data_clear/data0num'+str(batch*2)+'.npz',d1 = dataforlig, d2 = labelforlig)


