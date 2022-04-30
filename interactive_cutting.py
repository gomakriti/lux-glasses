import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, norm
from scipy.integrate import cumtrapz
import matplotlib.animation as ani
from utils import preprocessing, CS_interpolation, compute_cycles, get_cycles
from utils import get_accelerations_directions, normalize_length
from utils import count_steps
import pickle

in_file = "norm_walk1_fast.csv"
df = pd.read_csv(f"df_dataset/fast/{in_file}", index_col=False)
df = df[[
    "accx",
    "accy",
    "accz",
    "gyrx",
    "gyry",
    "gyrz",
    "timestamps"
    ]]
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


accelerations_norm = np.linalg.norm(accelerations, axis=1)
indexes = np.arange(len(accelerations_norm))
command = "0 -1"
samples = []
if os.path.isfile("raw_walk.pickle"):
    with open("raw_walk.pickle", "rb") as f:
        samples = pickle.load(f)
while command != "q":
    if " " in command:
        a, b = [int(val) for val in command.split()]
        plt.plot(indexes[a:b], accelerations_norm[a:b])
        plt.show()
    elif command == "s":
        print("saved data")
        samples.append([in_file, accelerations[a:b], angles_accelerations[a:b]])
        plt.plot(indexes[b:], accelerations_norm[b:])
        plt.show()
    print("write command")
    command = input()
with open("raw_walk.pickle", "wb") as f:
    pickle.dump(samples, f)

