import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import firwin, kaiserord, lfilter, argrelmin
import matplotlib.pyplot as plt

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


def preprocessing(signals):
    processed_s = CS_interpolation(signals)

    x = np.fft.fftfreq(len(processed_s), 1/200)
    plt.plot(x[:len(x)//2], np.abs(np.fft.fft(processed_s))[:len(x)//2])
    processed_s -= processed_s.mean()
    processed_s, pad = Fir_filtering(processed_s, 100, 60, 40)
    processed_s = processed_s[pad:-pad]
    processed_s = processed_s[400:]
    x = np.fft.fftfreq(len(processed_s), 1/200)
    plt.plot(x[:len(x)//2], np.abs(np.fft.fft(processed_s))[:len(x)//2])
    plt.yscale("log")
    plt.show()

    util_signal, pad = Fir_filtering(processed_s, 100, 60, 3)
    util_signal = util_signal[pad:-pad]
    minima_ind = argrelmin(util_signal)[0][1]
    
    print(f"Found minima at index {minima_ind}")
    actual_minima = np.argmin(
        processed_s[minima_ind-100:minima_ind+100]
    ) + minima_ind
    
    print(f"Found actual minima at index {actual_minima}")
    template = processed_s[actual_minima-100:actual_minima+100]
    plt.plot(template)
    plt.show()

    #getting correlation
    steps = [] 
    i = 100
    fig, axs = plt.subplots(1, 2)
    colors = plt.cm.get_cmap("Blues")(np.linspace(0, 1, 70))
    c_ind = 0
    cycles = []
    while i < len(processed_s) - 700:
        corrs = []
        for j in range(i, i+600):
            sample = processed_s[j-100:j+100]
            corr = corr_distance(template, sample)
            corrs.append(corr)
        corrs = np.array(corrs)
        # plt.plot(corrs)

        search_indexes = np.where(corrs < 0.5)
        searched_values = corrs[search_indexes]
        search_indexes = search_indexes[0]
        # plt.plot(search_indexes, searched_values)
        minima = argrelmin(searched_values)[0]
        min_indexes = search_indexes[minima] 
        min_indexes = parse_indexes(min_indexes, corrs)
        # plt.scatter(min_indexes, corrs[min_indexes])
        # plt.show()
        if i == 100:
            i = i + min_indexes[0]
            cycles.append(i + 400)
            min_indexes = min_indexes[1:] - min_indexes[0]
        steps.append(processed_s[i:i+min_indexes[1]])
        i += min_indexes[1]
        cycles.append(i +  400)
        template = 0.9 * template + 0.1 * processed_s[i-100:i+100]
        
        # plt.plot(steps[-1])
        # plt.show()
        axs[0].plot(steps[-1])
        axs[1].plot(template, c=colors[c_ind])
        c_ind +=1
    axs[1].plot(template, color="red")
    plt.show()
    return cycles

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
