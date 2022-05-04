import numpy as np
import pickle
from utils import preprocessing, CS_interpolation, compute_template, get_cycles
from utils import get_accelerations_directions, normalize_length
from utils import count_steps

with open("../dataset/raw_stairs.pickle", "rb") as f_stairs:
    stairs = pickle.load(f_stairs)

with open("../dataset/raw_walk.pickle", "rb") as f_walk:
    walk = pickle.load(f_walk)

with open("../dataset/test.pickle", "rb") as f_test:
    test = pickle.load(f_test)

def process_class(accelerometer, gyroscope, pressures, template_initialization=None):
    # the main signals are lists of shape: [n_signals, len(signal), channels]
    # note that the lists will not all have the same shape yet
    pressures = [preprocessing(pressure[:, 0]).reshape(-1, 1) for pressure in pressures]

    accelerometer = [np.vstack(
        [
            preprocessing(acc[:, 0]),
            preprocessing(acc[:, 1]),
            preprocessing(acc[:, 2]),
        ]
    ).T for acc in accelerometer]
    gyroscope = [np.vstack(
        [
            preprocessing(gyro[:, 0]),
            preprocessing(gyro[:, 1]),
            preprocessing(gyro[:, 2]),
        ]
    ).T for gyro in gyroscope]
    # the main signals are lists of shape: [n_signals, len(filtered_signal), channels]
    # note that the lists will not all have the same shape yet

    acc_norm = [np.linalg.norm(acc, axis=1) for acc in accelerometer]
    template = compute_template(acc_norm[0], template_initialization)
    cycles = [count_steps(acc, template) for acc in acc_norm]
    acceleration_cycles = [
        get_cycles(cycle, acc) for cycle, acc in zip(cycles, accelerometer)
    ]
    angles_acceleration_cycles = [
        get_cycles(cycle, angles) for cycle, angles in zip(cycles, gyroscope)
    ]
    pressure_cycles = [
        get_cycles(cycle, pressure) for cycle, pressure in zip(cycles, pressures)
    ]
    # the cycles variables are lists of shape: 
    # [n_signals, n_cycles, cycles_size, channels]
    # note that the lists will not all have the same shape yet
    stairs = [
        list(get_accelerations_directions(
           acc_cycle, angle_acc_cycle 
        )) + [pressure] for acc_cycle, angle_acc_cycle, pressure in zip(
            acceleration_cycles,
            angles_acceleration_cycles,
            pressure_cycles
        )
    ]
    # stairs will have shape [n_signals, 3 ,n_cycles, cycles_size, channels]
    # note that the lists will not all have the same shape yet

    return stairs, template

def merge_different_runs(signal):
    out_signals = [[], [], []]
    for a, g, p  in signal:
        out_signals[0] += list(normalize_length(a))
        out_signals[1] += list(normalize_length(g))
        out_signals[2] += list(normalize_length(p))
    return out_signals


up = stairs[::2]
down = stairs[1::2]
accelerometer_walk = [s["accelerometer"] for s in walk]
accelerometer_up = [s["accelerometer"] for s in up]
accelerometer_down = [s["accelerometer"] for s in down]
accelerometer_test = [s["accelerometer"] for s in test]
gyroscope_walk = [s["gyroscope"] for s in walk]
gyroscope_up = [s["gyroscope"] for s in up]
gyroscope_down = [s["gyroscope"] for s in down]
gyroscope_test = [s["gyroscope"] for s in test]
pressure_walk = [s["pressure"] for s in walk]
pressure_up = [s["pressure"] for s in up]
pressure_down = [s["pressure"] for s in down]
pressure_test = [s["pressure"] for s in test]

print("Processing normal walk")
samples_walk, template = process_class(
    accelerometer_walk,
    gyroscope_walk,
    pressure_walk
)
# normalizes cycles length and concatenates vectors
samples_walk = merge_different_runs(samples_walk)

print("Processing up the stairs")
samples_up, _ = process_class(
    accelerometer_up,
    gyroscope_up,
    pressure_up,
    template
)
# normalizes cycles length and concatenates vectors
samples_up = merge_different_runs(samples_up)

print("Processing down the stairs")
samples_down, _ = process_class(
    accelerometer_down,
    gyroscope_down,
    pressure_down,
    template
)
# normalizes cycles length and concatenates vectors
samples_down = merge_different_runs(samples_down)

print("Processing test data")
samples_test, _ = process_class(
    accelerometer_test,
    gyroscope_test,
    pressure_test,
    template
)

# concatenating along the channel direction
samples_up = np.concatenate(samples_up, axis=-1)
samples_down = np.concatenate(samples_down, axis=-1)
samples_walk = np.concatenate(samples_walk, axis=-1)

for i in range(len(samples_test)):
    for j in range(3):
        samples_test[i][j] = normalize_length(samples_test[i][j])
    samples_test[i] = np.concatenate(samples_test[i], axis=-1)

data_dict = {
    "walk": samples_walk,
    "up": samples_up,
    "down": samples_down,
    "test": samples_test,
}

with open("../dataset/data.pickle", "wb") as f:
    pickle.dump(data_dict, f)
