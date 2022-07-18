from calendar import c
from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan


class QuadControlSim:
    def __init__(self):
        self.sim_time = 10
        self.sim_step = 0.002
        self.sim_size = int(self.sim_time/self.sim_step)
        self.pointer = 0 

        self.quad_states = np.zeros((self.sim_size, 12))
        self.time= np.zeros((self.sim_size,))
        self.position_cmd = np.zeros((self.sim_size, 3)) 
        self.velocity_cmd = np.zeros((self.sim_size, 3)) 

        self.rate_cmd = np.zeros((self.sim_size, 3)) 
        self.attitude_cmd = np.zeros((self.sim_size, 3)) 

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])
        self.c = 0
        self.ez = np.array([0, 0, 1])
        self.R = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.ades = np.array([0,0,0]).T
        self.lastades = np.array([0,0,0]).T

    
    def run(self):
        for self.pointer in range(1, self.sim_size - 1):
            self.time[self.pointer] = self.pointer * self.sim_step            
            self.position_cmd[self.pointer] = [1,0,-1.5]
            psi_cmd = 0.0
            self.velocity_cmd[self.pointer] = self.position_controller(self.position_cmd[self.pointer])

            ##############################################################################################
            #pitch_roll_cmd, thrust_cmd = self.velocity_controller(self.velocity_cmd[self.pointer])
            ##############################################################################################
            xb, yb, zb, thrust_cmd = self.velocity_controller(self.velocity_cmd[self.pointer])
            #self.attitude_cmd[self.pointer] = np.append(pitch_roll_cmd,psi_cmd)
            self.rate_cmd[self.pointer] = self.attitude_controller(xb, yb, zb, thrust_cmd)
            torque_input = self.rate_controller(self.rate_cmd[self.pointer])
            #print(self.rate_cmd[self.pointer])

            self.quad_states[self.pointer+1] = self.quad_states[self.pointer] + self.sim_step*self.dynamics_model(thrust_cmd, torque_input)
            self.ez = self.R[:,2].T
            #print(yb)

        self.time[-1] = self.sim_time
        self.position_cmd[-1,:] = self.position_cmd[-2,:]
        self.velocity_cmd[-1,:] = self.position_controller(self.position_cmd[-1])
        self.c = thrust_cmd
        self.lastades = self.ades
     
        

    def dynamics_model(self,T,M):

        vx = self.quad_states[self.pointer,3]
        vy = self.quad_states[self.pointer,4]
        vz = self.quad_states[self.pointer,5]
        phi = self.quad_states[self.pointer,6]
        theta = self.quad_states[self.pointer,7]
        psi = self.quad_states[self.pointer,8]
        p = self.quad_states[self.pointer,9]
        q = self.quad_states[self.pointer,10]
        r = self.quad_states[self.pointer,11]

        R_d_angle = np.array([[1,tan(theta)*sin(phi),tan(theta)*cos(phi)],\
                             [0,cos(phi),-sin(phi)],\
                             [0,sin(phi)/cos(theta),cos(phi)/cos(theta)]])

        #x'y'z'(-phi, -theta, -psi)
        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])

        dp_dt = np.array([vx,vy,vz])
        #print(dp_dt)

        ######################################################################################
        #dv_dt = np.array([.0,.0,self.g]) + R_E_B.transpose()@np.array([.0,.0,T])
        dv_dt = np.array([.0,.0,self.g]) + self.R@np.array([.0,.0,T])  
        #print(self.R@np.array([.0,.0,T]))
        #print(np.array([.0,.0,T]))      
        ######################################################################################
        dangle_dt = R_d_angle@np.array([p,q,r])
        ######################################################################################

        dq_dt = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        #print(q)

        dx = np.concatenate((dp_dt, dv_dt, dangle_dt, dq_dt))

        w = np.array([[0, -r, q],[r, 0, -p],[-q, p, 0]])
        self.R = self.R + self.R@w*self.sim_step
        # print(w)
        #print(self.R)

        return dx

    def rate_controller(self,cmd):
        kp_p = 0.16 
        kp_q = 0.16
        kp_r = 0.28 
        error = cmd - self.quad_states[self.pointer,9:12]
        return np.array([kp_p*error[0],kp_q*error[1],kp_r*error[2]])*12

    def attitude_controller(self,xb, yb, zb, cmd):
        kp_phi = 2.5 
        kp_theta = 2.5 
        kp_psi = 2.5
        v1 = self.quad_states[self.pointer-1,3:6]
        v2 = self.quad_states[self.pointer,3:6]

        p = self.quad_states[self.pointer-1,9]
        q = self.quad_states[self.pointer-1,10]
        r = self.quad_states[self.pointer-1,11]

        xb = self.R[:,0].T
        yb = self.R[:,1].T
        zb = self.R[:,2].T
        R = np.column_stack((xb.T, yb.T, zb.T))
        R = np.matrix(R)
        ezz = np.matrix(self.ez).T
        #print(ezz)
        w = np.matrix([[0, -r, q],[r, 0, -p],[-q, p, 0]])
        #print(R.shape)
        xb = np.matrix(xb)
        yb = np.matrix(yb)
        zb = np.matrix(zb)
        # j = (cmd - self.c)/self.sim_step*zb.T
        # j = j + R*w*ezz*cmd 
                
        #j = np.matrix((self.ades -self.lastades) / self.sim_step /10000)

        j = np.matrix(self.ades-(v2-v1)/self.sim_step)*200
        #print(self.ades)

        wx = -yb*j.T/cmd
        wy = xb*j.T/cmd

        print(wy)

        psi = self.quad_states[self.pointer,8]
        psi0 = self.quad_states[self.pointer-1,8]
        xc = np.matrix([cos(psi),sin(psi),0])
        yc = np.matrix([-sin(psi),cos(psi),0])
        wy = np.array(wy)
        wwy = np.double(wy[0])
        
        wz = ((psi-psi0)/self.sim_step*xc*xb.T + wwy*yb.T)/np.linalg.norm(np.cross(yc,zb))
        wx = np.array(wx)

        wz = np.array(wz)
        #print(np.array([wx[0], wy[0], wz[0]]))
        #error = cmd - self.quad_states[self.pointer,6:9]
        #return np.array([kp_phi*error[0],kp_theta*error[1],kp_psi*error[2]])
        return np.array([wx[0], wy[0], wz[0]]).T

    def velocity_controller(self,cmd):
        kp_vx = 2
        kp_vy = 2
        kp_vz = 2
        #print(cmd)
        error = (cmd - self.quad_states[self.pointer,3:6])
        #print(error)
        ades = np.array([kp_vx*error[0],kp_vy*error[1],kp_vz*error[2]]) - self.g*np.array([0,0,1])
        a = ades
        #print(ades)

        # if self.pointer:
        #     a = self.quad_states[self.pointer, 3:6] - self.quad_states[self.pointer-1, 3:6]
        #     a = a/self.sim_step
        # else:
        #     a = np.array([0,0,0])
        psi = self.quad_states[self.pointer,8]
        #alpha = a + self.g*np.array([0,0,1])
        alpha = a
        #print(alpha)
        beta = alpha
        xc = np.array([cos(psi),sin(psi),0])
        yc = np.array([-sin(psi),cos(psi),0])

        # xb = np.cross(yc, alpha) 
        # #print(xb)       
        # xb = xb/np.linalg.norm(xb)
        # yb = np.cross(beta, xb)
        # yb = yb/np.linalg.norm(yb)
        # #print(xb)
        # zb = np.cross(xb,yb)

        zb = -ades/np.linalg.norm(ades) #######################
        xb = np.cross(yc, zb)
        xb = xb/np.linalg.norm(xb)
        yb = np.cross(zb, xb)

        #print(xb)


        # print(zb)
        #R = np.array([[cos(psi),sin(psi),0],[-sin(psi),cos(psi),0],[0,0,1]])
        #error = R@(cmd - self.quad_states[self.pointer,3:6])
        ezz = self.ez
        #print(ezz)
        #T = np.dot(zb, ades.T + self.g*np.array([[0,0,1]]).T)
        T = ezz[0]*ades[0] + ezz[1]*ades[1] + ezz[2]*ades[2]
        self.ades = ades
        # print(xb)
        print(T)
        #return np.array([kp_vy*error[1],kp_vx*error[0]]), T
        return xb, yb, zb, T

    def position_controller(self,cmd):
        kp_x = 0.7 
        kp_y = 0.7 
        kp_z = 0.7 
        error = cmd - self.quad_states[self.pointer,0:3]
        return np.array([kp_x*error[0],kp_y*error[1],kp_z*error[2]])


    def plot_states(self):
        fig1, ax1 = plt.subplots(4,3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0,0].plot(self.time,self.quad_states[:,0],label='real')
        ax1[0,0].plot(self.time,self.position_cmd[:,0],label='cmd')
        ax1[0,0].set_ylabel('x[m]')
        ax1[0,1].plot(self.time,self.quad_states[:,1])
        ax1[0,1].plot(self.time,self.position_cmd[:,1])
        ax1[0,1].set_ylabel('y[m]')
        ax1[0,2].plot(self.time,self.quad_states[:,2])
        ax1[0,2].plot(self.time,self.position_cmd[:,2])
        ax1[0,2].set_ylabel('z[m]')
        ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1,0].plot(self.time,self.quad_states[:,3])
        ax1[1,0].plot(self.time,self.velocity_cmd[:,0])
        ax1[1,0].set_ylabel('vx[m/s]')
        ax1[1,1].plot(self.time,self.quad_states[:,4])
        ax1[1,1].plot(self.time,self.velocity_cmd[:,1])
        ax1[1,1].set_ylabel('vy[m/s]')
        ax1[1,2].plot(self.time,self.quad_states[:,5])
        ax1[1,2].plot(self.time,self.velocity_cmd[:,2])
        ax1[1,2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2,0].plot(self.time,self.quad_states[:,6])
        ax1[2,0].plot(self.time,self.attitude_cmd[:,0])
        ax1[2,0].set_ylabel('phi[rad]')
        ax1[2,1].plot(self.time,self.quad_states[:,7])
        ax1[2,1].plot(self.time,self.attitude_cmd[:,1])
        ax1[2,1].set_ylabel('theta[rad]')
        ax1[2,2].plot(self.time,self.quad_states[:,8])
        ax1[2,2].plot(self.time,self.attitude_cmd[:,2])
        ax1[2,2].set_ylabel('psi[rad]')

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3,0].plot(self.time,self.quad_states[:,9])
        ax1[3,0].plot(self.time,self.rate_cmd[:,0])
        ax1[3,0].set_ylabel('p[rad/s]')
        ax1[3,1].plot(self.time,self.quad_states[:,10])
        ax1[3,1].plot(self.time,self.rate_cmd[:,1])
        ax1[3,0].set_ylabel('q[rad/s]')
        ax1[3,2].plot(self.time,self.quad_states[:,11])
        ax1[3,2].plot(self.time,self.rate_cmd[:,2])
        ax1[3,0].set_ylabel('r[rad/s]')

if __name__ == "__main__":
    quad = QuadControlSim()
    quad.run()
    quad.plot_states()
    plt.show()