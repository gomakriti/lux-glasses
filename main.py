import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, norm
from scipy.integrate import cumtrapz
import matplotlib.animation as ani
from utils import preprocessing, CS_interpolation, compute_template, get_cycles
from utils import get_accelerations_directions, normalize_length
from utils import count_steps

labels = (18, 22, 14, 26, 9, 9, 28)
for i in range(7, 8):
    print(f"Actual number of steps: {labels[i - 1]}")
    df = pd.read_csv(f"df_dataset/fast/sample{i}_fast.csv", index_col=False)
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

    template = compute_template(accelerations_norm)
    cycles = count_steps(accelerations_norm, template)

    # plt.plot(np.arange(len(accelerations))/200, accelerations_norm)
    # for x in cycles:
        # plt.axvline(x/200, c = "orange")
    # plt.show()

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
