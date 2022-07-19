from mimetypes import init
from tkinter.ttk import tclobjs_to_py
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan


class DroneControlSim:
    def __init__(self):
        self.xa = None
        self.xb = None
        self.xc = None
        self.ts = None

    def plan(self, waypointx, waypointy, waypointz):
        n_seg = len(waypointx)
        self.init_ts(n_seg)

    def init_ts(self, n_seg):
        self.ts = np.ones(n_seg)
        

if __name__ == "__main__":
    drone = DroneControlSim()
    drone.plan([0,1,0],[1,2,3],[-1,-1.5,0])
    