from mimetypes import init
from tkinter.ttk import tclobjs_to_py
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan
from math import factorial as fact


class DroneControlSim:
    def __init__(self):
        self.sim_time = 10
        self.sim_step = 0.002
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 12))
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.pointer = 0 
        self.T = 0

        self.ax = None
        self.ay = None
        self.az = None
        self.ts = 0
        self.ez = np.array([0, 0, 1])
        self.R = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.ades = 0

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])

        self.tss = None
        self.tsa = None
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
        self.tempi = 0
        self.endx = 0
        self.endy = 0
        self.endz = 0

        self.vxmax = 10
        self.vymax = 10
        self.vzmax = 10



    def drone_dynamics(self,T,M):
        x = self.drone_states[self.pointer,0]
        y = self.drone_states[self.pointer,1]
        z = self.drone_states[self.pointer,2]
        vx = self.drone_states[self.pointer,3]
        vy = self.drone_states[self.pointer,4]
        vz = self.drone_states[self.pointer,5]
        phi = self.drone_states[self.pointer,6]
        theta = self.drone_states[self.pointer,7]
        psi = self.drone_states[self.pointer,8]
        p = self.drone_states[self.pointer,9]
        q = self.drone_states[self.pointer,10]
        r = self.drone_states[self.pointer,11]

        R_d_angle = np.array([[1,tan(theta)*sin(phi),tan(theta)*cos(phi)],\
                             [0,cos(phi),-sin(phi)],\
                             [0,sin(phi)/cos(theta),cos(phi)/cos(theta)]])


        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])

        d_position = np.array([vx,vy,vz])
        d_velocity = np.array([.0,.0,self.g]) + R_E_B.transpose()@np.array([.0,.0,T])
        # print(d_velocity)
        d_angle = R_d_angle@np.array([p,q,r])
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        dx = np.concatenate((d_position,d_velocity,d_angle,d_q))

        self.R = R_E_B.T
        self.ez = self.R[:,2].T

        return dx 

    def run(self):
        for self.pointer in range(self.drone_states.shape[0]-1):
            self.time[self.pointer] = self.pointer * self.sim_step
            # print(self.tsa)
        
            if self.tempi < len(self.tsa)-1:
                if self.time[self.pointer] > self.tsa[self.tempi+1]: 
                    self.tempi = self.tempi + 1
                if self.tempi < len(self.tsa)-1:
                    self.ts = self.time[self.pointer] - self.tsa[self.tempi]
                    ts = self.ts
                    # print(ts)
                    t = np.array([ts**7, ts**6, ts**5, ts**4, ts**3, ts**2, ts, 1])
                    self.ax = self.polyx[0,8*self.tempi:8*(self.tempi+1)]
                    self.ay = self.polyy[0,8*self.tempi:8*(self.tempi+1)]
                    self.az = self.polyz[0,8*self.tempi:8*(self.tempi+1)]
                    # print(self.polyx)
                    xdes = np.dot(t, self.ax)
                    ydes = np.dot(t, self.ay)
                    zdes = np.dot(t, self.az)
                else:
                    self.ax = np.array([0,0,0,0,0,0,0,self.endx])
                    self.ay = np.array([0,0,0,0,0,0,0,self.endy])
                    self.az = np.array([0,0,0,0,0,0,0,self.endz])
            else:
                self.ax = np.array([0,0,0,0,0,0,0,self.endx])
                self.ay = np.array([0,0,0,0,0,0,0,self.endy])
                self.az = np.array([0,0,0,0,0,0,0,self.endz])


            psi_cmd = 0.0
    
            self.position_cmd[self.pointer] = [xdes, ydes, zdes]
            # print(xdes, ydes, zdes)
            self.velocity_cmd[self.pointer] = self.position_controller(self.position_cmd[self.pointer])            
            # self.velocity_cmd[self.pointer] = [0.0,0.0,-1.0]
            pitch_roll_cmd,thrust_cmd = self.velocity_controller(self.velocity_cmd[self.pointer])
            self.attitude_cmd[self.pointer] = np.append(pitch_roll_cmd,psi_cmd)
            #self.attitude_cmd[self.pointer] = [1,0,0]
            self.rate_cmd[self.pointer] = self.attitude_controller(self.attitude_cmd[self.pointer])
            # self.rate_cmd[self.pointer] = [1,0,0]
            M = self.rate_controller(self.rate_cmd[self.pointer])
            # thrust_cmd = -10 * self.m

            self.drone_states[self.pointer+1] = self.drone_states[self.pointer] + self.sim_step*self.drone_dynamics(thrust_cmd,M)
            
        self.time[-1] = self.sim_time



    def rate_controller(self,cmd):
        kp_p = 0.016*2 
        kp_q = 0.016*2 
        kp_r = 0.028*2 
        error = cmd - self.drone_states[self.pointer,9:12]
        return np.array([kp_p*error[0],kp_q*error[1],kp_r*error[2]])

    def attitude_controller(self,cmd):
        kp_phi = 3.5
        kp_theta = 3.5
        kp_psi = 3.5
        psi = self.drone_states[self.pointer,8]
        yc = np.array([-sin(psi),cos(psi),0])

        xb = self.R[:,0].T
        yb = self.R[:,1].T
        zb = self.R[:,2].T

        ts = self.ts
        tj = np.array([210*ts**4, 120*ts**3, 60*ts**2, 24*ts, 6, 0, 0, 0])
        xj = np.dot(tj, self.ax)
        yj = np.dot(tj, self.ay)
        zj = np.dot(tj, self.az)
        j = np.zeros(3)
        j[0] = xj
        j[1] = yj
        j[2] = zj
        # print(j)
        ta = np.array([42*ts**5, 30*ts**4, 20*ts**3, 12*ts**2, 6*ts, 2, 0, 0])
        aref = np.zeros(3)
        aref[0] = np.dot(ta, self.ax)
        aref[1] = np.dot(ta, self.ay)
        aref[2] = np.dot(ta, self.az)
        alpha = aref - self.g*np.array([0,0,1])
        xb = np.cross(alpha,yc)
        # print(xb)
        xb = xb / np.linalg.norm(xb)
        yb = np.cross(xb,alpha)
        yb = yb / np.linalg.norm(yb)
        # xb = -xb
        zb = np.cross(xb, yb)
        # print(zb)

        c = np.dot(zb, alpha)

        # print(c*zb)
        
        wx = -np.dot(yb,j)/c
        wy = np.dot(xb,j)/c
        wz = wy*np.dot(yc,zb)/np.linalg.norm(np.cross(yc,zb))
        
        # wx = -np.dot(yb,j)/self.T
        # wy = np.dot(xb,j)/self.T
        # wz = wy*np.dot(yc,zb)/np.linalg.norm(np.cross(yc,zb))
        # print(self.T)

        error = cmd - self.drone_states[self.pointer,6:9]
        # print(error)
        # print(np.array([kp_phi*error[0]+wx,kp_theta*error[1]+wy,kp_psi*error[2]+wz]))
        # return np.array([kp_phi*error[0]+wx,kp_theta*error[1]+wy,kp_psi*error[2]+wz])
        return np.array([kp_phi*error[0],kp_theta*error[1],kp_psi*error[2]])+np.array([wx, wy, wz])


    def velocity_controller(self,cmd):
        kp_vx = -0.35
        kp_vy = 0.35
        kp_vz = 3.5
        ts = self.ts
        tv = np.array([7*ts**6, 6*ts**5, 5*ts**4, 4*ts**3, 3*ts**2, 2*ts, 1, 0])
        ta = np.array([42*ts**5, 30*ts**4, 20*ts**3, 12*ts**2, 6*ts, 2, 0, 0])
        vref = np.zeros(3)
        vref[0] = np.dot(tv, self.ax)
        vref[1] = np.dot(tv, self.ay)
        vref[2] = np.dot(tv, self.az)
        # print(vref)
        error = (vref - self.drone_states[self.pointer,3:6])
        aref = np.zeros(3)
        aref[0] = np.dot(ta, self.ax)
        aref[1] = np.dot(ta, self.ay)
        aref[2] = np.dot(ta, self.az)
        #print(error)
        # print(aref)
        ades = cmd + np.array([kp_vx*error[0],kp_vy*error[1],kp_vz*error[2]]) + aref - self.g*np.array([0,0,1])
        ezz = self.ez
        T = ezz[0]*ades[0] + ezz[1]*ades[1] + ezz[2]*ades[2]
        self.ades = ades
        self.T = T
        # print(T)

        psi = self.drone_states[self.pointer,8]
        R = np.array([[cos(psi),sin(psi),0],[-sin(psi),cos(psi),0],[0,0,1]])
        error = R@(cmd - self.drone_states[self.pointer,3:6])
        return np.array([kp_vy*error[1],kp_vx*error[0]]), T

    def position_controller(self,cmd):
        kp_x = 0.7
        kp_y = 0.7 
        kp_z = 1.2 

        error = cmd - self.drone_states[self.pointer,0:3]
        return np.array([kp_x*error[0],kp_y*error[1],kp_z*error[2]])


    def plot_states(self):
        fig1, ax1 = plt.subplots(4,3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0,0].plot(self.time,self.drone_states[:,0],label='real')
        ax1[0,0].plot(self.time,self.position_cmd[:,0],label='cmd')
        ax1[0,0].set_ylabel('x[m]')
        ax1[0,1].plot(self.time,self.drone_states[:,1])
        ax1[0,1].plot(self.time,self.position_cmd[:,1])
        ax1[0,1].set_ylabel('y[m]')
        ax1[0,2].plot(self.time,self.drone_states[:,2])
        ax1[0,2].plot(self.time,self.position_cmd[:,2])
        ax1[0,2].set_ylabel('z[m]')
        ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1,0].plot(self.time,self.drone_states[:,3])
        ax1[1,0].plot(self.time,self.velocity_cmd[:,0])
        ax1[1,0].set_ylabel('vx[m/s]')
        ax1[1,1].plot(self.time,self.drone_states[:,4])
        ax1[1,1].plot(self.time,self.velocity_cmd[:,1])
        ax1[1,1].set_ylabel('vy[m/s]')
        ax1[1,2].plot(self.time,self.drone_states[:,5])
        ax1[1,2].plot(self.time,self.velocity_cmd[:,2])
        ax1[1,2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2,0].plot(self.time,self.drone_states[:,6])
        ax1[2,0].plot(self.time,self.attitude_cmd[:,0])
        ax1[2,0].set_ylabel('phi[rad]')
        ax1[2,1].plot(self.time,self.drone_states[:,7])
        ax1[2,1].plot(self.time,self.attitude_cmd[:,1])
        ax1[2,1].set_ylabel('theta[rad]')
        ax1[2,2].plot(self.time,self.drone_states[:,8])
        ax1[2,2].plot(self.time,self.attitude_cmd[:,2])
        ax1[2,2].set_ylabel('psi[rad]')

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3,0].plot(self.time,self.drone_states[:,9])
        ax1[3,0].plot(self.time,self.rate_cmd[:,0])
        ax1[3,0].set_ylabel('p[rad/s]')
        ax1[3,1].plot(self.time,self.drone_states[:,10])
        ax1[3,1].plot(self.time,self.rate_cmd[:,1])
        ax1[3,0].set_ylabel('q[rad/s]')
        ax1[3,2].plot(self.time,self.drone_states[:,11])
        ax1[3,2].plot(self.time,self.rate_cmd[:,2])
        ax1[3,0].set_ylabel('r[rad/s]')

    def plan(self, waypointx, waypointy, waypointz):
        self.endx = waypointx[-1]
        self.endy = waypointy[-1]
        self.endz = waypointz[-1]
        # print(self.endx, self.endy, self.endz)

        self.n_seg = len(waypointx)-1
        self.init_ts(waypointx, waypointy, waypointz)
        self.calQ()
        self.calM()
        self.calC()
        self.calR()   

        self.polyx = self.calcpoly(waypointx) 
        self.polyy = self.calcpoly(waypointy)
        self.polyz = self.calcpoly(waypointz)

        # print(self.polyz)


    def init_ts(self, waypointx, waypointy, waypointz):
        # to be ++++++++++++++++++
        self.tss = np.ones(self.n_seg)
        for i in range(len(self.tss)):
            t1 = abs(waypointx[i+1]-waypointx[i]) / self.vxmax
            t2 = abs(waypointy[i+1]-waypointy[i]) / self.vymax
            t3 = abs(waypointz[i+1]-waypointz[i]) / self.vzmax
            self.tss[i] = max(t1,max(t2,t3))
        self.tsa = np.zeros(self.n_seg+1)
        for i in range(1, len(self.tsa)):
            self.tsa[i] = self.tsa[i-1] + self.tss[i-1]

        # print(self.tsa)

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
        return polyx

if __name__ == "__main__":
    drone = DroneControlSim()
    # drone.plan([0,2,3],[0,-1.5,-2],[0,-1,-2])
    drone.plan([0,5,10],[0,10,5],[0,-5,-10])
    # drone.plan([0,5],[0,10],[0,-5])

    drone.run()
    drone.plot_states()
    plt.show()