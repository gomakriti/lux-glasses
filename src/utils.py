import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import firwin, kaiserord, lfilter, argrelmin
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def CS_interpolation(signal, sampling_freq=50, final_sampling_freq = 200):

    x = np.linspace(0, (len(signal) - 1) / sampling_freq, len(signal))
    cs = CubicSpline(x, signal)

    new_x = np.linspace(
        0,
        (len(signal) - 1) / sampling_freq,
        len(signal) * (final_sampling_freq // sampling_freq)
    )
    interplotated_signal = cs(new_x)

    return interplotated_signal

def fir_filtering(signal, nyq_rate, ripple_db, cutoff_hz):
    width = 5 / nyq_rate
    N, beta = kaiserord(ripple_db, width)
    taps = firwin(N, cutoff_hz/nyq_rate, window=("kaiser", beta))
    pad = len(taps) // 2
    filtered_signal = lfilter(taps, 1, signal)
    return filtered_signal, pad


def preprocessing(signal, nyq_rate=100, ripple_db=60, cutoff_hz=40):

    interplotated_signal = CS_interpolation(signal)

    filtered_signal, pad = fir_filtering(
        interplotated_signal, 
        nyq_rate,
        ripple_db,
        cutoff_hz
    )
    filtered_signal = filtered_signal[pad:-pad]

    return filtered_signal

def compute_template(
    signal,
    template_initialization = None,
    first_sample_height=1200,
    corr_thr = 0.5,
    window_size = 400,
    template_size = 200
):

    first_index = np.where(signal > first_sample_height)[0][0]
    # cutting everything before the first peak since it is just transitory information
    processed_signal = signal[first_index:].copy()
    # normalizing the signal by its mean (somehow considerably improves detection)
    processed_signal -= processed_signal.mean()

    # fir filtering with low cutoff frequency to find out the where to begin 
    # the template search
    nyquist_rate = 100
    ripple_db = 60
    cutoff_hz = 3
    filtered_signal, pad = fir_filtering(
        processed_signal,
        nyquist_rate,
        ripple_db,
        cutoff_hz
    )
    filtered_signal = filtered_signal[pad:-pad]
    # the second minima will be the considered place (taking the second for stability)
    minima_ind = argrelmin(filtered_signal)[0][1]
    
    # searching for the actual minima in the original signal in a 1 second window
    actual_minima = np.argmin(
        processed_signal[minima_ind:minima_ind+template_size]
    ) + minima_ind
    
    # if there ins't an already preinitialized template then select it in a 1 seconda 
    # window around the minima that was recently found
    half_template = template_size // 2
    if template_initialization is None:
        template = processed_signal[
            actual_minima-half_template:actual_minima+half_template
        ]
    else:
        template = template_initialization.copy()

    # Starting to compute the correlation function
    # inital window center
    i = 100
    while i < len(processed_signal) - window_size - half_template:
        #correlation vector
        corrs = []
        # window center going from i to i + 400 (i.e. 2 seconds) to find the first match
        for j in range(i, i+window_size):
            #selecting the sample to be matched to the template
            sample = processed_signal[j-half_template:j+half_template]
            # checking if this is actually walking data or if the person is just 
            # standing still, this is needed because the noise when standng still
            # can sometime lead to uncorrect matches
            # this is done by checking the amplitude of the considered sample
            delta = lambda x: np.max(x) - np.min(x)
            if delta(sample) < delta(template) / 10:
                corr = 1
            else:
                corr = corr_distance(template, sample)
            # appending the chosen correlation distance
            corrs.append(corr)
        corrs = np.array(corrs)

        # searching for points where the template has a possible match
        search_indexes = np.where(corrs < corr_thr)
        # getting the values of the correlation in correspondence of the selected 
        # indexes
        searched_values = corrs[search_indexes]
        # getting the array of indexes, the first dimension was just needed to use it
        # for slicing
        search_indexes = search_indexes[0]
        # getting the first minima in the correlation values
        minima = argrelmin(searched_values)[0]
        # getting the actual index in the correlation vector
        # corresponding to the first minima
        min_indexes = search_indexes[minima] 
        # discarding the minima if it is not the absolute minima in a small window
        # (used to avoid detection of local minima)
        min_indexes = parse_indexes(min_indexes, corrs)

        # if needed to handle the very first match since it doesn't have an end to the 
        # step
        if i == 100:
            #moving forward the window
            i = i + min_indexes[0]
            min_indexes = min_indexes[1:] - min_indexes[0]
        # if not enough minima were found just stop the cycle i.e. this works only until
        # the person stops walking
        if len(min_indexes) < 2:
            break
        i += min_indexes[1]
        template = 0.9 * template + 0.1 * processed_signal[i-100:i+100]
        
    return template

def count_steps(signal, template, thr = 0.45):
    template_size = len(template)
    # as done for the template computation function we subtract the signal mean
    signal -= signal.mean()
    # computing the correlation
    corrs = []
    for i in range(len(signal) - template_size):
        #getting the sample to be matched
        sample = signal[i:i+template_size]

        # as previously lets remove the stationary parts to avoid fake matches
        delta = lambda x: np.max(x) - np.min(x)
        if delta(sample) < delta(template) / 2:
            corr = 1
        else:
            corr = corr_distance(template, sample)
        corrs.append(corr)
    corrs = np.array(corrs)

    #same as above finding the indexes of the minima and removing bad ones
    search_indexes = np.where(corrs < thr)
    searched_values = corrs[search_indexes]
    minima = argrelmin(searched_values)[0]
    search_indexes = search_indexes[0]
    min_indexes = search_indexes[minima] 
    min_indexes = parse_indexes(min_indexes, corrs)

    # this function should be used only if you are sure that a person kept walking, 
    # in this way you can make sure that even if a step is not detected a good
    # estimate of the step is still obtained
    new_indexes = correct_missing_indexes(min_indexes)
    print("Number of steps:",len(new_indexes))
    # plt.figure()
    # plt.hlines(thr, 0, len(signal))
    # plt.plot(corrs)
    # plt.xlim([-100, len(signal) + 100])

    # plt.show()
    return new_indexes

def correct_missing_indexes(min_indexes):
    # getting the median step length
    median = np.median(np.diff(min_indexes))
    start_index = min_indexes[0]
    indexes = [start_index]
    for min_index in min_indexes[1:]:
        # checking if the step interval is greater than 1.5 times the median, if so
        # the interval is divided in round(len(interval) / median intervals)
        if  (min_index - start_index) > 1.5 * median:
            n_missed = int(np.round((min_index - start_index) / median))
            for j in range(1, n_missed):
                indexes.append(
                    int(start_index + j * np.round((min_index - start_index) / n_missed))
                )
        indexes.append(min_index)
        start_index = min_index
    return np.array(indexes)

def parse_indexes(min_indexes, corrs, half_window=20):
    # checking if there is another minimum in a window of size 2*half_window, if so
    # ignore the current minimum
    parsed_indexes = []
    for index in min_indexes:
        if min(half_window, index) == np.argmin(
            corrs[max(index-half_window, 0):min(index+half_window, len(corrs))]
        ):
            parsed_indexes.append(index)
    return np.array(parsed_indexes)

# correlation distance computation
def corr_distance(u, v):
    u -= np.mean(u)
    v -= np.mean(v)
    return 1 - np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)

# given the cycles indexes this function extracts the actual signals
def get_cycles(cycles, signal):
    pieces = []
    for i in range(len(cycles) - 1):
        pieces.append(signal[cycles[i]:cycles[i+1]].reshape(
            (cycles[i+1] - cycles[i], -1)
        ))
    return pieces

def get_accelerations_directions(acceleration_cycles, gyro_cycles):
    # normalizing accelerometer and gyroscope data w.r.t. a common reference system
    new_accelerations = []
    new_gyroscopes = []
    for i, (cycle, gyro_cycle) in enumerate(zip(acceleration_cycles, gyro_cycles)):
        # getting the average acceleration mean in the cycle i.e. the first axis
        first_axis = cycle.mean(axis=0)

        # normalizing the vector that will represent the gravity versor
        gravity = first_axis.reshape(1, 3) / np.linalg.norm(first_axis)
        normalized_acc = []
        normalized_gyr = []

        # getting the component along the gravity versor
        normalized_acc.append(cycle@gravity.T)
        normalized_gyr.append(gyro_cycle@gravity.T)

        # removing the component along the gravity axis
        main_direction_acc = normalized_acc[0] * gravity
        main_direction_gyr = normalized_gyr[0] * gravity
        planar_direction_acc = cycle - main_direction_acc
        planar_direction_gyr = gyro_cycle - main_direction_gyr
        
        # computing PCA along the resulting plane to obtain the remaining directions
        pca_acc = PCA(2).fit(planar_direction_acc)
        pca_gyr = PCA(2).fit(planar_direction_gyr)

        #getting the components
        acc_y, acc_z = pca_acc.components_
        gyr_y, gyr_z = pca_gyr.components_
        # this is needed to make sure that the principal vectors look all in a 
        # similar direction, for this reason they are oriented towards the first
        # chosen versors
        if i == 0:
            acc_y0, acc_z0 = acc_y, acc_z
            gyr_y0, gyr_z0 = gyr_y, gyr_z
        acc_y = normalize_principal_vector(acc_y, acc_y0)
        acc_z = normalize_principal_vector(acc_z, acc_z0)
        gyr_y = normalize_principal_vector(gyr_y, gyr_y0)
        gyr_z = normalize_principal_vector(gyr_z, gyr_z0)
        
        # adding the remaining components
        normalized_acc.append(cycle@acc_y.reshape(3, 1))
        normalized_acc.append(cycle@acc_z.reshape(3, 1))
        normalized_gyr.append(gyro_cycle@gyr_y.reshape(3, 1))
        normalized_gyr.append(gyro_cycle@gyr_z.reshape(3, 1))
        new_accelerations.append(np.concatenate(normalized_acc, axis=1))
        new_gyroscopes.append(np.concatenate(normalized_gyr, axis=1))

    return new_accelerations, new_gyroscopes

def normalize_principal_vector(v, v0):
    # if the versors look more or less in the same direction then keep v as it is
    # otherwise swap it
    if v.reshape(1, 3)@v0.reshape((3, 1)) < 0:
        return -v
    else:
        return v

# reinterpolates the data so all samples have the same length
def normalize_length(cycles, target_sample_length = 200):

    normalized_cycles = []

    for cycle in cycles:

        cycle_vect = []
        cycle = np.array(cycle)

        for vec in cycle.T:

            x = np.arange(len(vec))
            cs = CubicSpline(x, vec)

            new_x = np.linspace(0, len(vec), target_sample_length)
            interpolated = cs(new_x)
            cycle_vect.append(interpolated)

        normalized_cycles.append(np.vstack(cycle_vect).T)

    return normalized_cycles

