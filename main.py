import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, norm
from scipy.integrate import cumtrapz
import matplotlib.animation as ani
from utils import preprocessing, CS_interpolation, compute_cycles, get_cycles
from utils import get_accelerations_directions, normalize_length


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
accelerations = np.vstack([
    preprocessing(accelerations[:, 0]),
    preprocessing(accelerations[:, 1]),
    preprocessing(accelerations[:, 2])
]).T
angles_accelerations = np.vstack([
    preprocessing(angles_accelerations[:, 0]),
    preprocessing(angles_accelerations[:, 1]),
    preprocessing(angles_accelerations[:, 2])
]).T
accelerations_norm = np.linalg.norm(accelerations, axis=1)
angles_accelerations_norm = np.linalg.norm(angles_accelerations, axis=1)
cycles = compute_cycles(accelerations_norm)
plt.plot(np.arange(len(accelerations))/200, accelerations_norm)
for x in cycles:
    plt.axvline(x/200, c = "orange")
plt.show()

acceleration_cycles = get_cycles(cycles, accelerations)
angles_acceleration_cycles = get_cycles(cycles, angles_accelerations)
acc, gyr = get_accelerations_directions(
    acceleration_cycles,
    angles_acceleration_cycles
)
n_cycles = 10
for j in range(0, n_cycles, n_cycles):
    fig, axs = plt.subplots(2, 3)
    for i in range(3):
        axs[0, i].plot(np.vstack(acc[j:j+n_cycles])[:, i])
        axs[0, i].set_title("oriented")
        axs[1, i].plot(np.vstack(acceleration_cycles[j:j+n_cycles])[:, i])
        axs[1, i].set_title("original")
        for x in cycles[:n_cycles]:
            axs[0, i].axvline((x - cycles[0]), c = "orange")
            axs[1, i].axvline((x - cycles[0]), c = "orange")
    plt.show()

acc = normalize_length(acc)
fig, axs = plt.subplots(3, 1)
for a in acc[:10]:
    for i in range(3):
        axs[i].plot(a[:, i] - a[:, i].mean())
plt.show()
gyr = normalize_length(gyr)
