# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:43:48 2023

@author: chenqi Tian   
"""
from numba import jit
import numba
import numpy as np
import math
from numba import cuda
# 光速
c = 300000  # km/s
@jit(nopython=True)
def computingZns(maxlen, studygrids, u_list, stationlist, sample):
    Zn = np.zeros((maxlen, len(studygrids)))
    for rn in range(len(u_list)):
        for lm in range(len(studygrids)):
            rmn= np.sqrt((stationlist[rn][0]-studygrids[lm][0])**2+
                                      (stationlist[rn][1]-studygrids[lm][1])**2+
                                      (stationlist[rn][2]-studygrids[lm][2])**2)  
            for k in range(maxlen):
                un = u_list[rn][::-1]
                if k-int(rmn/sample/c) < 0:
                    s = 0
                else:
                    s = un[k-int(rmn/c/sample)]
                Zn[k][lm] = Zn[k][lm] + s
    return Zn

@jit(nopython=True)
def computingZn(maxlen, studygrids, u_list, stationlist, sample, traveltime):
    Zn = np.zeros((maxlen, len(studygrids)))
    for rn in range(len(u_list)):
        for k in range(maxlen):
            for lm in range(len(studygrids)):
                un = u_list[rn][::-1]
                if k-int(traveltime[lm][rn]/sample) < 0:
                    s = 0
                else:
                    s = un[k-int(traveltime[lm][rn]/sample)]
                Zn[k][lm] = Zn[k][lm] + s
    return Zn

@jit(nopython = True)
def computingZnFDs(maxlen,sample,studygrids,U_conjlist,stationlist):
    fs = 1/sample
    Zn = np.zeros((maxlen,len(studygrids)),dtype=numba.complex64)  
    for rn,Un in enumerate(U_conjlist):
        for lm,rm in enumerate(studygrids):
            rmn= np.sqrt((stationlist[rn][0]-studygrids[lm][0])**2+
                                      (stationlist[rn][1]-studygrids[lm][1])**2+
                                      (stationlist[rn][2]-studygrids[lm][2])**2)  
            for k in range(maxlen):    
                x = -2j*math.pi*(k-1)*fs/maxlen*rmn/c
                s = np.e**x*Un[k]
                Zn[k][lm] = Zn[k][lm] + s
    return Zn

@jit(nopython = True)
def computingZnFD(maxlen,sample,studygrids,U_conjlist,stationlist,traveltime):
    fs = 1/sample
    Zn = np.zeros((maxlen,len(studygrids)),dtype=numba.complex64)  
    for rn,Un in enumerate(U_conjlist):
        for k in range(maxlen):
            for lm,rm in enumerate(studygrids):
                x = -2j*math.pi*(k-1)*fs/maxlen*traveltime[lm][rn]
                s = np.e**x*Un[k]
                Zn[k][lm] = Zn[k][lm] + s
    return Zn



@cuda.jit
def computingZn_GPU(Zn,maxlen,studygrids,u_list,stationlist,sample,traveltime): 
    lm = cuda.blockIdx.x
    rn = cuda.threadIdx.x
    for k in range(maxlen):
        un=u_list[rn][::-1]
        if k-int(traveltime[lm][rn]/sample) < 0:
            s = 0
        else:
            s = un[k-int(traveltime[lm][rn]/sample)]
        cuda.atomic.add(Zn[k], lm, s)
@cuda.jit
def computingZns_GPU(Zn,maxlen,studygrids,u_list,stationlist,sample): 
    lm = cuda.blockIdx.x
    rn = cuda.threadIdx.x
    rmn= math.sqrt((stationlist[rn][0]-studygrids[lm][0])**2+
                          (stationlist[rn][1]-studygrids[lm][1])**2+
                          (stationlist[rn][2]-studygrids[lm][2])**2)  
    for k in range(maxlen):
        un=u_list[rn][::-1]
        if k-int(rmn/sample/c) < 0:
            s = 0
        else:
            s = un[k-int(rmn/sample/c)]
        cuda.atomic.add(Zn[k], lm, s)
        
@cuda.jit 
def computingZnFD_GPU(Zn,maxlen,sample,studygrids,U_conjlist,stationlist,traveltime):
    fs = 1/sample
    lm = cuda.blockIdx.x
    rn = cuda.threadIdx.x
    for k in range(maxlen):
        x = -2j*math.pi*(k-1)*fs/maxlen*traveltime[lm][rn]
        s = np.e**x*U_conjlist[rn][k]
        cuda.atomic.add(Zn[k].real, lm, s.real)
        cuda.atomic.add(Zn[k].imag, lm, s.imag)


        
@cuda.jit  
def computingZnFDs_GPU(Zn,maxlen,sample,studygrids,U_conjlist,stationlist):
    fs = 1/sample
    lm = cuda.blockIdx.x
    rn = cuda.threadIdx.x
    rmn= math.sqrt((stationlist[rn][0]-studygrids[lm][0])**2+
                          (stationlist[rn][1]-studygrids[lm][1])**2+
                          (stationlist[rn][2]-studygrids[lm][2])**2)  
    for k in range(maxlen):

        x = -2j*math.pi*(k-1)*fs/maxlen*rmn/c
        s = np.e**x*U_conjlist[rn][k]
        cuda.atomic.add(Zn[k].real, lm, s.real)
        cuda.atomic.add(Zn[k].imag, lm, s.imag)
        