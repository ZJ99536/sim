from ctypes import wintypes
from tkinter import W
from tkinter.ttk import tclobjs_to_py
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan


class DroneControlSim:
    def __init__(self):
        self.sim_time = 2
        self.sim_step = 0.002
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 12))
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.pointer = 0 
        self.T = 0

        self.ax = np.array([-6,   21,  -25.2,  10.5,   0,    0,    0,    0])
        # self.ax = np.array([0,0,0,0,0,0,0,5])
        # self.ax = np.array([-8.383750e+00,  2.876125e+01, -3.352125e+01,  1.334375e+01,  0, 0, 0, 0])
        self.ay = np.array([-6,   21,  -25.2,  10.5,   0,    0,    0,    0])
        # self.ay = np.array([0,0,0,0,0,0,0,5])
        # self.az = np.array([0,0,0,0,0,0,0,-5])

        self.az = np.array([4,  -14,   16.8,  -7,   0,    0,    0,    0])

        # self.az = np.array([0,0,0,0,-0.01,0.03,0,-1.5])
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

        self.flag = 1
        


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
        print(R_E_B)
        d_velocity = np.array([.0,.0,self.g]) + R_E_B.transpose()@np.array([.0,.0,T])
        print(R_E_B)
        # print(d_velocity)
        d_angle = R_d_angle@np.array([p,q,r])
        # print(d_angle)
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        dx = np.concatenate((d_position,d_velocity,d_angle,d_q))

        self.R = R_E_B.T
        self.ez = self.R[:,2].T

        return dx 

    def run(self):
        for self.pointer in range(self.drone_states.shape[0]-1):
            self.time[self.pointer] = self.pointer * self.sim_step
            psi_cmd = 0.0
            self.ts = self.pointer * self.sim_step
            ts = self.ts
            t = np.array([ts**7, ts**6, ts**5, ts**4, ts**3, ts**2, ts, 1])
            xdes = np.dot(t, self.ax)
            ydes = np.dot(t, self.ay)
            zdes = np.dot(t, self.az)
            if self.time[self.pointer]>8 and self.flag==1:
                ts = 8
                t = np.array([ts**7, ts**6, ts**5, ts**4, ts**3, ts**2, ts, 1])
                xdes = np.dot(t, self.ax)
                ydes = np.dot(t, self.ay)
                zdes = np.dot(t, self.az)
                self.ax = np.array([0,0,0,0,0,0,0,xdes])
                self.ay = np.array([0,0,0,0,0,0,0,ydes])
                self.az = np.array([0,0,0,0,0,0,0,zdes])
                self.flag = 0
            elif self.time[self.pointer]>8 and self.flag==0:
                ts = 8
                t = np.array([ts**7, ts**6, ts**5, ts**4, ts**3, ts**2, ts, 1])
                self.ax = np.array([0,0,0,0,0,0,0,xdes])
                self.ay = np.array([0,0,0,0,0,0,0,ydes])
                self.az = np.array([0,0,0,0,0,0,0,zdes])
                xdes = np.dot(t, self.ax)
                ydes = np.dot(t, self.ay)
                zdes = np.dot(t, self.az)
                            
            
            self.position_cmd[self.pointer] = [xdes, ydes, zdes]
            self.velocity_cmd[self.pointer] = self.position_controller(self.position_cmd[self.pointer])

            
            # self.velocity_cmd[self.pointer] = [0.0,0.0,-1.0]
            pitch_roll_cmd,thrust_cmd = self.velocity_controller(self.velocity_cmd[self.pointer])
            self.attitude_cmd[self.pointer] = np.append(pitch_roll_cmd,psi_cmd)

            #self.attitude_cmd[self.pointer] = [1,0,0]
            self.rate_cmd[self.pointer] = self.attitude_controller(self.attitude_cmd[self.pointer])

            # self.rate_cmd[self.pointer] = [1,0,0]
            M = self.rate_controller(self.rate_cmd[self.pointer])
            # thrust_cmd = -10 * self.m
            ts = self.ts
            tv = np.array([7*ts**6, 6*ts**5, 5*ts**4, 4*ts**3, 3*ts**2, 2*ts, 1, 0])
            ta = np.array([42*ts**5, 30*ts**4, 20*ts**3, 12*ts**2, 6*ts, 2, 0, 0])
            vref = np.zeros(3)
            vref[0] = np.dot(tv, self.ax)
            vref[1] = np.dot(tv, self.ay)
            vref[2] = np.dot(tv, self.az)
            self.velocity_cmd[self.pointer] = vref
            aref = np.zeros(3)
            aref[0] = np.dot(ta, self.ax)
            aref[1] = np.dot(ta, self.ay)
            aref[2] = np.dot(ta, self.az)
            self.attitude_cmd[self.pointer] = aref
            self.drone_states[self.pointer+1] = self.drone_states[self.pointer] + self.sim_step*self.drone_dynamics(thrust_cmd,M)
            
        self.time[-1] = self.sim_time



    def rate_controller(self,cmd):
        kp_p = 1
        kp_q = 1 
        kp_r = 2 
        # xb = self.R[:,0].T
        # yb = self.R[:,1].T
        # zb = self.R[:,2].T
        # psi = self.drone_states[self.pointer,8]
        # yc = np.matrix([-sin(psi),cos(psi),0])

        # ts = self.ts
        # tj = np.array([210*ts**4, 120*ts**3, 60*ts**2, 24*ts, 6, 0, 0, 0])
        # xj = np.dot(tj, self.ax)
        # yj = np.dot(tj, self.ay)
        # zj = np.dot(tj, self.az)
        # j = np.zeros(3)
        # j[0] = xj
        # j[1] = yj
        # j[2] = zj        
        # cdot = np.dot(zb,j)

        # tsn = np.array([840*ts**3, 360*ts**2, 120*ts, 24, 0, 0, 0, 0])
        # xs = np.dot(tsn, self.ax)
        # ys = np.dot(tsn, self.ay)
        # zs = np.dot(tsn, self.az)
        # s = np.zeros(3)
        # s[0] = xs
        # s[1] = ys
        # s[2] = zs 

        # wydot = (np.dot(xb,s) - 2*cdot*cmd[1] -self.T*cmd[0]*cmd[1])/self.T
        # wxdot = (-np.dot(yb,s) - 2*cdot*cmd[0] + self.T*cmd[1]*cmd[2])/self.T
        # wzdot = -(cmd[0]*cmd[1]*np.dot(yc,yb)+cmd[0]*cmd[2]*np.dot(yc,yb)-wydot*np.dot(yc,zb))/np.linalg.norm(np.cross(yc,zb))

        # print(zb)
        # cmd[0] = -cmd[0]

        error = cmd - self.drone_states[self.pointer,9:12]
        return np.array([kp_p*error[0],kp_q*error[1],kp_r*error[2]])
        # return np.array([wxdot, wydot, wzdot])

    def attitude_controller(self,cmd):
        kp_phi = 0 
        kp_theta = 0 
        kp_psi = 0
        psi = self.drone_states[self.pointer,8]
        # yc = np.matrix([-sin(psi),cos(psi),0])

        yc = np.array([-sin(psi),cos(psi),0])

        # xb = np.cross(yc, alpha) 
        # #print(xb)       
        # xb = xb/np.linalg.norm(xb)
        # yb = np.cross(beta, xb)
        # yb = yb/np.linalg.norm(yb)
        # #print(xb)
        # zb = np.cross(xb,yb)

        ts = self.ts
        tj = np.array([210*ts**4, 120*ts**3, 60*ts**2, 24*ts, 6, 0, 0, 0])
        xj = np.dot(tj, self.ax)
        yj = np.dot(tj, self.ay)
        zj = np.dot(tj, self.az)
        j = np.zeros(3)
        j[0] = xj
        j[1] = yj
        j[2] = zj
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
        # print(j)

        error = cmd - self.drone_states[self.pointer,6:9]

        # print(wx, wy, wz)
        # print(np.array([kp_phi*error[0]+wx,kp_theta*error[1]+wy,kp_psi*error[2]+wz]))

        
        return np.array([kp_phi*error[0]+wx,kp_theta*error[1]+wy,kp_psi*error[2]+wz])

    def velocity_controller(self,cmd):
        kp_vx = 0
        kp_vy = 0
        kp_vz = 0
        ts = self.ts
        tv = np.array([7*ts**6, 6*ts**5, 5*ts**4, 4*ts**3, 3*ts**2, 2*ts, 1, 0])
        ta = np.array([42*ts**5, 30*ts**4, 20*ts**3, 12*ts**2, 6*ts, 2, 0, 0])
        vref = np.zeros(3)
        vref[0] = np.dot(tv, self.ax)
        vref[1] = np.dot(tv, self.ay)
        vref[2] = np.dot(tv, self.az)
        error = (vref - self.drone_states[self.pointer,3:6])
        aref = np.zeros(3)
        aref[0] = np.dot(ta, self.ax)
        aref[1] = np.dot(ta, self.ay)
        aref[2] = np.dot(ta, self.az)
        #print(error)
        ades = aref - self.g*np.array([0,0,1])
        ezz = self.ez
        print(ezz)
        T = ezz[0]*ades[0] + ezz[1]*ades[1] + ezz[2]*ades[2]
        self.ades = ades
        self.T = T
        # print(T)
        # print(aref)

        psi = self.drone_states[self.pointer,8]
        R = np.array([[cos(psi),sin(psi),0],[-sin(psi),cos(psi),0],[0,0,1]])
        error = R@(cmd - self.drone_states[self.pointer,3:6])
        # if error[1] > 0:
        #     a1 = min(kp_vy*error[1],1.5)
        # else:
        #     a1 = max(kp_vy*error[1],-1.5)
        # if error[0] > 0:
        #     a2 = min(kp_vx*error[0],1.5)
        # else:
        #     a2 = max(kp_vx*error[0],-1.5)
        # if T < -15:
        #     T = 15
        return np.array([kp_vy*error[1], kp_vx*error[0]]), T

    def position_controller(self,cmd):
        kp_x = 0
        kp_y = 0 
        kp_z = 0 

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
        

if __name__ == "__main__":
    drone = DroneControlSim()
    drone.run()
    drone.plot_states()
    plt.show()