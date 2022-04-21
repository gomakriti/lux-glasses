import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, norm
from scipy.integrate import cumtrapz
import matplotlib.animation as ani
from utils import preprocessing


def rotate(vec, angle):
    phi, theta, psi = angle
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    rvec = Rz @ Ry @ Rx @ vec
    return rvec.T

def get_anim_func_angles(positions, velocity):

    min_val = np.min(positions)
    max_val = np.max(positions)
    delta = (max_val - min_val)/10
    x = 0
    y = 2

    def get_angles(i):
        colors = ["red", "blue", "green"]
        if i*5 > len(positions):
            quit()
        plt.clf()
        plt.xlim([min_val - delta, max_val + delta])
        plt.ylim([min_val - delta, max_val + delta])
        plt.xlabel("x")
        plt.ylabel("z")
        plt.scatter([positions[:5*i, x]], [positions[:5*i, y]], cmap="viridis", c=np.arange(5*i))
        plt.plot([positions[5*i, x], positions[5*i, x] + velocity[5*i, x] * 20], [positions[5*i, y], positions[5*i, y] + velocity[5*i, y] * 20], c="blue")
        plt.title(f"{i}")
    return get_angles

df = pd.read_csv("dataset_luxottica/fast/out5_fast.csv", index_col=False)
df = df[[
    "accx",
    "accy",
    "accz",
    "gyrx",
    "gyry",
    "gyrz",
    "magx",
    "magy",
    "magz",
    "roll",
    "pitch",
    "yaw",
    "timestamps"
    ]]
df["timestamps"] -= np.min(df["timestamps"].min())
aps = 15
time_interval = 0.3 / 15
angle = np.zeros((1, 3))
position = np.zeros((1, 3))
velocity = np.zeros((1, 3))
angle_velocity = np.zeros((1, 3))

positions = []
velocities = []
magnetics = []

accelerations = df[[ 
    "accx",
    "accy",
    "accz",
]].to_numpy()
angles_accelerations = df[[ 
    "gyrx",
    "gyry",
    "gyrz",
]].to_numpy()
magnetic_fields = df[[ 
    "magx",
    "magy",
    "magz",
]].to_numpy()
angles = df[[
    "pitch" ,
    "roll" ,
    "yaw",
]].to_numpy()

angles -= angles[0].reshape((1, 3))
accelerations = np.linalg.norm(accelerations, axis=1)
angles_accelerations = np.linalg.norm(angles_accelerations, axis=1)
preprocessing(accelerations)
