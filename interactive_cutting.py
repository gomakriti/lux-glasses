import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, norm
from scipy.integrate import cumtrapz
import matplotlib.animation as ani
from utils import preprocessing, CS_interpolation, compute_template, get_cycles
from utils import get_accelerations_directions, normalize_length
from utils import count_steps
import pickle
from scipy.interpolate import CubicSpline

in_file = "test2"
df = pd.read_csv(f"df_dataset/fast/{in_file}_fast.csv", index_col=False)
df_slow = pd.read_csv(f"df_dataset/slow/{in_file}_slow.csv", index_col=False)
df = df[[
    "accx",
    "accy",
    "accz",
    "gyrx",
    "gyry",
    "gyrz",
    "timestamps",
    ]]
pressure = df_slow[["pressure"]]
x = np.linspace(0, len(pressure) - 1, len(pressure))
cs = CubicSpline(x, pressure)
new_x = np.linspace(0, len(pressure) - 1, len(pressure) * 5)
pressure = cs(new_x)
df["timestamps"] -= np.min(df["timestamps"].min())

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

out_filename = "df_dataset/test.pickle"


accelerations_norm = np.linalg.norm(accelerations, axis=1)
indexes = np.arange(len(accelerations_norm))
command = "0 -1"
samples = []
if os.path.isfile(out_filename):
    with open(out_filename, "rb") as f:
        samples = pickle.load(f)
while command != "q":
    if " " in command:
        a, b = [int(val) for val in command.split()]
        plt.plot(indexes[a:b], accelerations_norm[a:b])
        plt.show()
    elif command == "s":
        print("saved data")
        samples.append({
            "filename": in_file,
            "accelerometer": accelerations[a:b],
            "gyroscope": angles_accelerations[a:b],
            "pressure": pressure[a:b]
        })
        plt.plot(indexes[b:], accelerations_norm[b:])
        plt.show()
    print("write command")
    command = input()
print(len(samples), len(samples[0]["accelerometer"]))
with open(out_filename, "wb") as f:
    pickle.dump(samples, f)

