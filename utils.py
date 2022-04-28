import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import firwin, kaiserord, lfilter, argrelmin
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def CS_interpolation(signals):
    x = np.linspace(0, (len(signals) - 1)/ 50, len(signals))
    cs = CubicSpline(x, signals)
    new_x = np.linspace(0, (len(signals) - 1)/50, len(signals) * 4)
    new_interp = cs(new_x)
    return new_interp

def Fir_filtering(signals, nyq_rate, ripple_db, cutoff_hz):
    width=5/nyq_rate
    N, beta = kaiserord(ripple_db, width)
    taps = firwin(N, cutoff_hz/nyq_rate,window=("kaiser", beta))
    pad = len(taps) // 2
    filtered_x = lfilter(taps, 1, signals)
    return filtered_x, pad


def preprocessing(signal):
    processed_s = CS_interpolation(signal)

    x = np.fft.fftfreq(len(processed_s), 1/200)
    # plt.plot(x[:len(x)//2], np.abs(np.fft.fft(processed_s))[:len(x)//2])
    processed_s, pad = Fir_filtering(processed_s, 100, 60, 40)
    processed_s = processed_s[pad:-pad]
    x = np.fft.fftfreq(len(processed_s), 1/200)
    # plt.plot(x[:len(x)//2], np.abs(np.fft.fft(processed_s))[:len(x)//2])
    # plt.yscale("log")
    # plt.show()
    return processed_s

def compute_cycles(signal):
    processed_s = signal[400:].copy()
    processed_s -= processed_s.mean()

    util_signal, pad = Fir_filtering(processed_s, 100, 60, 3)
    util_signal = util_signal[pad:-pad]
    minima_ind = argrelmin(util_signal)[0][1]
    
    actual_minima = np.argmin(
        processed_s[minima_ind:minima_ind+200]
    ) + minima_ind
    
    template = processed_s[actual_minima-100:actual_minima+100]

    #getting correlation
    steps = [] 
    i = 100
    # fig, axs = plt.subplots(1, 2)
    # colors = plt.cm.get_cmap("Blues")(np.linspace(0, 1, 70))
    c_ind = 0
    cycles = []
    while i < len(processed_s) - 700:
        corrs = []
        for j in range(i, i+600):
            sample = processed_s[j-100:j+100]
            delta = lambda x: np.max(x) - np.min(x)
            if delta(sample) < delta(template) / 10:
                corr = 1
            else:
                corr = corr_distance(template, sample)
            corrs.append(corr)
        corrs = np.array(corrs)

        search_indexes = np.where(corrs < 0.5)
        searched_values = corrs[search_indexes]
        search_indexes = search_indexes[0]
        minima = argrelmin(searched_values)[0]
        min_indexes = search_indexes[minima] 
        min_indexes = parse_indexes(min_indexes, corrs)
        # plt.scatter(min_indexes, corrs[min_indexes])
        # plt.show()
        if i == 100:
            i = i + min_indexes[0]
            cycles.append(i + 400)
            min_indexes = min_indexes[1:] - min_indexes[0]
        if len(min_indexes) < 2:
            break
        steps.append(processed_s[i:i+min_indexes[1]])
        i += min_indexes[1]
        cycles.append(i +  400)
        template = 0.9 * template + 0.1 * processed_s[i-100:i+100]
        
        # axs[0].plot(steps[-1])
        # axs[1].plot(template, c=colors[c_ind])
        c_ind +=1
    # axs[1].plot(template, color="red")
    # plt.show()
    return template

def count_steps(signal, template):
    signal -= signal.mean()
    corrs = []
    for i in range(len(signal) - len(template)):
        sample = signal[i:i+200]
        delta = lambda x: np.max(x) - np.min(x)
        if delta(sample) < delta(template) / 10:
            corr = 1
        else:
            corr = corr_distance(template, sample)
        corrs.append(corr)
    thr = 0.4
    corrs = np.array(corrs)
    search_indexes = np.where(corrs < thr)
    searched_values = corrs[search_indexes]
    minima = argrelmin(searched_values)[0]
    search_indexes = search_indexes[0]
    min_indexes = search_indexes[minima] 
    min_indexes = parse_indexes(min_indexes, corrs)
    # plt.vlines(min_indexes + 100, -1000, 1000, color="orange")
    # plt.ylim([-300, 800])
    print("Number of steps:",len(min_indexes))
    # plt.figure()
    # plt.hlines(thr, 0, len(signal))
    # plt.plot(corrs)
    # plt.xlim([-100, 4000])
    # plt.show()
    return min_indexes


def parse_indexes(min_indexes, corrs):
    parsed_indexes = []
    for index in min_indexes:
        if min(20, index) == np.argmin(
            corrs[max(index-20, 0):min(index+20, len(corrs))]
        ):
            parsed_indexes.append(index)
    return np.array(parsed_indexes)

def corr_distance(u, v):
    u -= np.mean(u)
    v -= np.mean(v)
    return 1 - np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)

def get_cycles(cycles, signal):
    pieces = []
    for i in range(len(cycles) - 1):
        pieces.append(signal[cycles[i]:cycles[i+1]])
    return pieces

def get_accelerations_directions(acceleration_cycles, gyro_cycles):
    new_accelerations = []
    new_gyroscopes = []
    for i, (cycle, gyro_cycle) in enumerate(zip(acceleration_cycles, gyro_cycles)):
        first_axis = cycle.mean(axis=0)
        gravity = first_axis.reshape(1, 3) / np.linalg.norm(first_axis)
        normalized_acc = []
        normalized_gyr = []
        normalized_acc.append(cycle@gravity.T)
        normalized_gyr.append(gyro_cycle@gravity.T)
        main_direction_acc = normalized_acc[0] * gravity
        main_direction_gyr = normalized_gyr[0] * gravity
        planar_direction_acc = cycle - main_direction_acc
        planar_direction_gyr = gyro_cycle - main_direction_gyr
        pca_acc = PCA(2).fit(planar_direction_acc)
        pca_gyr = PCA(2).fit(planar_direction_gyr)
        acc_y, acc_z = pca_acc.components_
        gyr_y, gyr_z = pca_gyr.components_
        if i == 0:
            acc_y0, acc_z0 = acc_y, acc_z
            gyr_y0, gyr_z0 = gyr_y, gyr_z
        acc_y = normalize_principal_vector(acc_y, acc_y0)
        acc_z = normalize_principal_vector(acc_z, acc_z0)
        gyr_y = normalize_principal_vector(gyr_y, gyr_y0)
        gyr_z = normalize_principal_vector(gyr_z, gyr_z0)
        normalized_acc.append(cycle@acc_y.reshape(3, 1))
        normalized_acc.append(cycle@acc_z.reshape(3, 1))
        normalized_gyr.append(gyro_cycle@gyr_y.reshape(3, 1))
        normalized_gyr.append(gyro_cycle@gyr_z.reshape(3, 1))
        new_accelerations.append(np.concatenate(normalized_acc, axis=1))
        new_gyroscopes.append(np.concatenate(normalized_gyr, axis=1))
    return new_accelerations, new_gyroscopes

def normalize_principal_vector(v, v0):
    if v.reshape(1, 3)@v0.reshape((3, 1)) < 0:
        return -v
    else:
        return v

def normalize_length(cycles):
    normalized_cycles = []
    for cycle in cycles:
        cycle_vect = []
        for vec in cycle.T:
            x = np.arange(len(vec))
            cs = CubicSpline(x, vec)
            new_x = np.linspace(0, len(vec), 200)
            new_interp = cs(new_x)
            cycle_vect.append(new_interp)
        normalized_cycles.append(np.vstack(cycle_vect).T)
    return normalized_cycles

