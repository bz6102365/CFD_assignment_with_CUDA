import matplotlib.pyplot as plt
import numpy as np
from ctypes import *
import math

def init(x):
    if x<=0.2:
        return 0.0
    elif x<=0.5:
        return math.sin((x-0.2)*10*math.pi)
    elif x<=0.7:
        return 7.5*(x-0.5)
    else:
        return -1.0
"""
def init(x):
    if x<=0.5:
        return 0.0
    return 1.0

"""
nums=1000

class solverInterface:
    def __init__(self,dllPath,label,linestyle="-"):
        self.linestyle=linestyle
        self.label=label
        self.dll=windll.LoadLibrary(dllPath)
        initCondition=[init(x) for x in np.linspace(0,1,nums+1)]
        self.cpoint=(c_float*(nums+1))()
        for i in range(nums+1):
            self.cpoint[i]=c_float(initCondition[i])
        self.ptr=pointer(self.cpoint)
        c_a=c_float(0.2)
        c_cell_num=c_int(nums)
        c_totalTime=c_float(1.0)
        c_dt=c_float(0.0001)
        self.dll.initKernel(self.ptr,c_a,c_cell_num,c_totalTime,c_dt)

    def step(self):
        self.dll.stepKernel()

    def getdata(self):
        self.dll.getData(self.ptr)
        return list(self.cpoint)
    
    def plot_cur_step(self):
        plt.plot(np.linspace(0,1,nums+1),self.getdata(),label=self.label,linestyle=self.linestyle)

root=r"D:\cfd\bin2\cudaSolution2."
solverUW=solverInterface(root+"1.dll","upper-wind")
solverBW=solverInterface(root+"2.dll","Beam-Warming",)
solverTVD=solverInterface(root+"3.dll","TVD",)
solverENORK3=solverInterface(root+"4.dll","ENO-RK3",)

for t in range(100):
    for _ in range(100):
        solverUW.step()
        solverBW.step()
        solverTVD.step()
        solverENORK3.step()
    plt.cla()
    solverUW.plot_cur_step()
    solverBW.plot_cur_step()
    solverTVD.plot_cur_step()
    solverENORK3.plot_cur_step()

    exact=[init(x-0.2*t*0.01) for x in np.linspace(0,1,nums+1)]
    plt.plot(np.linspace(0,1,nums+1),exact,label="exact")

    plt.legend()
    plt.ylim([-1.5,1.5])
    plt.pause(0.1)

plt.cla()
solverUW.plot_cur_step()
solverBW.plot_cur_step()
solverTVD.plot_cur_step()
solverENORK3.plot_cur_step()
exact=[init(x-0.2) for x in np.linspace(0,1,nums+1)]
plt.plot(np.linspace(0,1,nums+1),exact,label="exact")
plt.legend()
plt.ylim([-2.0,2.0])
plt.show()
