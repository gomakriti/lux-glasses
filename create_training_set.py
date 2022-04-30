import numpy as np
import pickle
from utils import preprocessing, CS_interpolation, compute_cycles, get_cycles
from utils import get_accelerations_directions, normalize_length
from utils import count_steps

with open("raw_stairs.pickle", "rb") as f:
    stairs = pickle.load(f)

with open("raw_walk.pickle", "rb") as f:
    walk = pickle.load(f)

def process_class(accelerometer, gyroscope, template_initialization=None):
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
    acc_norm = [np.linalg.norm(acc, axis=1) for acc in accelerometer]
    template = compute_cycles(acc_norm[0], template_initialization)
    cycles = [count_steps(acc, template) for acc in acc_norm]
    acceleration_cycles = [
        get_cycles(cycle, acc) for cycle, acc in zip(cycles, accelerometer)
    ]
    angles_acceleration_cycles = [
        get_cycles(cycle, angles) for cycle, angles in zip(cycles, gyroscope)
    ]
    print(sum([len(x) for x in acceleration_cycles]))

    stairs = [
        get_accelerations_directions(
           acc_cycle, angle_acc_cycle 
        ) for acc_cycle, angle_acc_cycle in zip(
            acceleration_cycles,
            angles_acceleration_cycles
        )
    ]
    return stairs, template

print(len(stairs))
up = stairs[::2]
down = stairs[1::2]
accelerometer_walk = [s[1] for s in walk]
accelerometer_up = [s[1] for s in up]
accelerometer_down = [s[1] for s in up]
gyroscope_walk = [s[2] for s in walk]
gyroscope_up = [s[2] for s in up]
gyroscope_down = [s[2] for s in up]
print("time for normal walk")
samples, template = process_class(accelerometer_walk, gyroscope_walk)
print("time for up the stairs")
samples, _ = process_class(accelerometer_up, gyroscope_up, template)
print("time for down the stairs")
samples, _ = process_class(accelerometer_down, gyroscope_down, template)
