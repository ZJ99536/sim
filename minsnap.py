from mimetypes import init
from tkinter.ttk import tclobjs_to_py
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan
from math import factorial as fact



class DroneControlSim:
    def __init__(self):
        self.tss = None
        self.n_seg = 0
        self.n_order = 7
        self.Q = None
        self.M = None
        self.C = None
        self.Rp = None
        self.Rpp = None
        self.Rfp = None
        self.polyx = None
        self.polyy = None
        self.polyz = None

    def plan(self, waypointx, waypointy, waypointz):
        self.n_seg = len(waypointx)-1
        self.init_ts()
        self.calQ()
        self.calM()
        self.calC()
        self.calR()   

        self.polyx = self.calcpoly(waypointx) 
        self.polyy = self.calcpoly(waypointy)
        self.polyz = self.calcpoly(waypointz)


    def init_ts(self):
        # to be ++++++++++++++++++
        self.tss = np.ones(self.n_seg)

    def calQ(self):
        n_seg = self.n_seg
        n_order = self.n_order
        self.Q = np.zeros((n_seg * (n_order + 1), n_seg * (n_order + 1)))
        ts = self.tss
        for n in range(n_seg):
            for i in range(4, n_order + 1):
                for j in range(4, n_order + 1):
                    self.Q[n*(n_order+1) + i, n*(n_order+1) + j] = fact(i) * fact(j) / (fact(i-4)*fact(j-4)*(i+j-7)) * ts[n]**(i+j-7)
                    # print(n*(n_order+1) + i)
        # print(self.Q)

    def calM(self):
        # to be ++++++++++++++++++
        n_seg = self.n_seg
        n_order = self.n_order
        self.M = np.zeros((n_seg * (n_order + 1), n_seg * (n_order + 1)))
        ts = self.tss
        for i in range(n_seg):
            self.M[8*i+0, 8*i+7] = 1
            self.M[8*i+1, 8*i+6] = 1
            self.M[8*i+2, 8*i+5] = 2
            self.M[8*i+3, 8*i+4] = 6
            for j in range(n_order + 1):
                self.M[8*i+4, 8*i+j] = ts[i]**(n_order-j)
                self.M[8*i+5, 8*i+j] = (n_order-j)*ts[i]**(n_order-j-1)
                self.M[8*i+6, 8*i+j] = (n_order-j-1)*(n_order-j)*ts[i]**(n_order-j-2)
                self.M[8*i+7, 8*i+j] = (n_order-j-2)*(n_order-j-1)*(n_order-j)*ts[i]**(n_order-j-3)
        # print(self.M)  

    def calC(self):
        # to be ++++++++++++++++++
        n_seg = self.n_seg
        n_order = self.n_order
        self.C = np.zeros((n_seg * (n_order + 1), 4 * (n_seg + 1))) 
        for i in range(4):
            self.C[i, i] = 1
            self.C[8*n_seg-i-1, n_seg+6-i] = 1
        for j in range(1, n_seg):
            self.C[8 * (j-1) + 4, j+3] = 1;
            self.C[8 * (j-1) + 5, n_seg+7+(j-1)*3] = 1;
            self.C[8 * (j-1) + 6, n_seg+8+(j-1)*3] = 1;
            self.C[8 * (j-1) + 7, n_seg+9+(j-1)*3] = 1;
            self.C[8 * j, j+3] = 1;
            self.C[8 * j+1, n_seg+7+(j-1)*3] = 1;
            self.C[8 * j+2, n_seg+8+(j-1)*3] = 1;
            self.C[8 * j+3, n_seg+9+(j-1)*3] = 1;
        # print(self.C.shape)
        self.C = self.C.transpose()
        # print(self.C.shape)

    def calR(self):
        n_seg = self.n_seg
        n_order = self.n_order
        M_inv = np.matrix(np.linalg.inv(self.M))
        M_inv_T = np.matrix(M_inv.T)
        C = np.matrix(self.C)
        Q = np.matrix(self.Q)
        Ct = np.matrix(self.C.T)
        self.Rp = np.array(C * M_inv_T * Q * M_inv * Ct)
        # print(self.Rp)

        self.Rpp = self.Rp[n_seg+7:4*n_seg+4,n_seg+7:4*n_seg+4]
        self.Rfp = self.Rp[0:n_seg+7,n_seg+7:4*n_seg+4]
        # print(self.Rpp)
        # print(self.Rfp)

    def calcpoly(self,waypoint):
        df = np.zeros(self.n_seg + 7) 
        df[0] = waypoint[0] 
        df[4:self.n_seg+4] = waypoint[1:self.n_seg+1]
        
        if self.n_seg > 1:
            R_pp_inv = np.matrix(np.linalg.inv(self.Rpp))
            R_fp_T = np.matrix(self.Rfp).transpose()
            dF = np.matrix(df).transpose()
            # print(R_fp_T.shape)
            dp = -np.array(R_pp_inv*R_fp_T*dF).T
            # print(dp.shape)
            d = np.zeros(4*(self.n_seg +1))
            d[0:self.n_seg+7] = df[0:self.n_seg+7]
            d[self.n_seg+7:4*(self.n_seg+1)] = dp[0:3*(self.n_seg-1)]
        else:
            d = df
        # print(d)

        M_inv = np.matrix(np.linalg.inv(self.M))
        Ct = np.matrix(self.C).transpose()
        dm = np.matrix(d).transpose()
        polyx = np.array((M_inv * Ct * dm).transpose())
        # print(polyx.shape)

        # t = np.zeros(int(2/0.02)+1)
        # timex = np.zeros(int(2/0.02)+1)
        # x = np.zeros(int(2/0.02)+1)
        # for i in range(len(t)):
        #     t[i] = i*0.02
        #     timex[i] = i*0.02
        # for i in range(len(t)):
        #     if t[i] < 1:
        #         tt = np.array([t[i]**7,t[i]**6,t[i]**5,t[i]**4,t[i]**3,t[i]**2,t[i],1])
        #         poly = polyx[0,0:8]
        #         x[i] = np.dot(tt,poly)
        #     else:
        #         t[i] = t[i] - 1
        #         tt = np.array([t[i]**7,t[i]**6,t[i]**5,t[i]**4,t[i]**3,t[i]**2,t[i],1])
        #         poly = polyx[0,8:16]
        #         x[i] = np.dot(tt,poly)
        # print(timex)
        # plt.plot(timex,x,label='x')
        # plt.show()
        return polyx

    

if __name__ == "__main__":
    drone = DroneControlSim()
    drone.plan([0,1.5,2],[1,2],[-1,-1.5])
    