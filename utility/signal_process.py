"""This module is used to modify filters & remove the noise of the
raw data."""

# Imports
import numpy as np
import scipy.signal

from glob import glob
from scipy.signal import lfilter, butter
from collections import deque
import sys


def modify_filters(fft, freq_bins, percentage):
    """This function will filter the signal of the fast fourier
    transform by a percentage of the frequency with the maximum
    magnitude.

    Args:
        fft (numpy.ndarray): This is the fast fourier transform of an
                             audio waveform.
        freq_bins (numpy.ndarray): These are the bins of each frequency
                                   of the FFT.
        percentage (float): This is the percentage at which to filter
                            the FFT.

    Returns:
        numpy.ndarray: This is the filtered FFT which has each freqency
                       equal to or below the calculated threshold set to
                       zero.
    """
    threshold = percentage * (2 * np.abs(fft[0 : len(fft) // 2]) / len(freq_bins)).max()
    filtered_fft = fft.copy()
    filtered_fft_magnitude = np.abs(filtered_fft)
    filtered_fft_magnitude = 2 * filtered_fft_magnitude / len(freq_bins)
    filtered_fft[filtered_fft_magnitude <= threshold] = 0
    return filtered_fft


def preprocess_to_frequency_domain(raw_signal_array, sample_rate):
    """This method converts the signal from a raw signal to a
    preprocessed signal in the frequency domain. It will detrend the
    signal before returning the fast fourier transform of the detrended
    signal for further processing. This function serves as an
    intermediate preprocessing first step.

    Args:
        raw_signal_array (numpy.ndarray): This is the raw signal in
                                          numpy array format.
        sample_rate (int): This is the rate at which the data was
                           sampled.

    Returns:
        fft (numpy.ndarray): This is the transformed signal in the
                             frequency domain.
        freq_bins (numpy.ndarray): This is the bins of frequencies
                                pertaining to the fast fourier
                                transform.
    """
    detrended_signal = scipy.signal.detrend(raw_signal_array)
    fft = np.fft.fft(detrended_signal)
    freq_bins = np.arange(
        start=0, stop=(sample_rate // 2), step=(sample_rate / len(fft))
    )
    return fft, freq_bins


def identify_potential_initial_spikes(amplitude_array, return_local_maximum=True):
    """This function searches for peak amplitudes that may be initial
    neural spiking activity. This function is extended to filter the
    local maximum or minimum spiking activity. This is used to identify
    second or third spikes as well.

    Args:
        amplitude_array (numpy.ndarray): This contains an array of
                                         amplitudes of neural signal.
        return_local_maximum (bool, optional): This defines the logic of
                                               the returned values. If
                                               True, the values will be
                                               the local maximums of the
                                               amplitude array. When
                                               False,the returned list
                                               will be local minimums.

    Returns:
        list: This is a list of boolean values that indicate whether a
        point is a local maximum with respect to the next and previous
        amplitudes. If return_local_maximum is set to False, then the
        returned list contains information of local minimums instead.
    """
    if len(amplitude_array) < 3:
        if len(amplitude_array) == 0:
            return ValueError("Length of amplitude array must be greater than 0")
        elif len(amplitude_array) == 1:
            return [True]
        else:
            if return_local_maximum:
                if amplitude_array[0] < amplitude_array[1]:
                    return [False, True]
                else:
                    return [True, False]
            else:
                if amplitude_array[0] < amplitude_array[1]:
                    return [True, False]
                else:
                    return [False, True]
    else:
        if return_local_maximum:
            local_maximum_list = []
            for idx, val in enumerate(amplitude_array[0:-1]):
                if idx == 0:
                    if amplitude_array[idx + 1] < val:
                        local_maximum_list.append(True)
                    else:
                        local_maximum_list.append(False)
                    continue
                if (amplitude_array[idx - 1] < val) and (
                    val > amplitude_array[idx + 1]
                ):
                    local_maximum_list.append(True)
                else:
                    local_maximum_list.append(False)
            if amplitude_array[-1] > amplitude_array[-2]:
                local_maximum_list.append(True)
            else:
                local_maximum_list.append(False)
            return local_maximum_list
        else:
            local_minimum_list = []
            for idx, val in enumerate(amplitude_array[0:-1]):
                if idx == 0:
                    if amplitude_array[idx + 1] > val:
                        local_minimum_list.append(True)
                    else:
                        local_minimum_list.append(False)
                    continue
                if (amplitude_array[idx - 1] > val) and (
                    val < amplitude_array[idx + 1]
                ):
                    local_minimum_list.append(True)
                else:
                    local_minimum_list.append(False)
            if amplitude_array[-1] < amplitude_array[-2]:
                local_minimum_list.append(True)
            else:
                local_minimum_list.append(False)
            return local_minimum_list


def estimate_noise_floor(amplitude_array, window_size=10):
    """This function will estimate the noise floor. The amplitude array
    must be at least of length of the window size or a single value.

    Args:
        amplitude_array (numpy.ndarray): Array of amplitudes with which
                                         to derive the noise floor.

        window_size (int, optional): This is the width of the window
                                     used to calculate a rolling median
                                     average.

    Return:
        noise_floor_estimate (numpy.ndarray): This is the estimate of the
                                           noise floor.
    """
    if len(amplitude_array) == 0:
        raise ValueError("Length of amplitude array must be greater than 0")
    elif len(amplitude_array) == 1:
        noise_floor_estimate = np.array(np.sqrt(np.abs(amplitude_array) ** 2))
        return noise_floor_estimate
    else:
        if len(amplitude_array) < window_size:
            window_size = len(amplitude_array)
        power_of_filtered_data = np.abs(amplitude_array) ** 2

        rolling_median_array = []
        for index in range(0, len(power_of_filtered_data), 1):
            current_median = np.median(
                power_of_filtered_data[index : index + window_size]
            )
            rolling_median_array.append(current_median)

        rolling_median_array = np.array(rolling_median_array)

        noise_floor_estimate = np.sqrt(rolling_median_array)

        return noise_floor_estimate


def detect_neural_spikes(neural_data):
    """This function detects spikes in real-time.
    It returns an array of indices of spike locations.

    Args:
        neural_data (array): This is the array of amplitudes for each
                             point of time of the neural data.

    Returns:
        spike_train_time_index_list (list): This is the array inclusive
                                            of amplitudes of spikes at
                                            each specific point in the
                                            initial time array.
                                            Non-spike points have been
                                            replaced with amplitudes of
                                            zero value.
    """
    noise_floor_window = 5
    initial_first_point_of_spike_detected = False
    second_point_of_spike_detected = False
    third_point_of_spike_detected = False
    spike_train_time_index_list = []

    for current_time_index, amplitude in enumerate(neural_data):
        # Estimate the noise floor
        if current_time_index < noise_floor_window:
            current_noise_floor_estimate_list = estimate_noise_floor(
                [neural_data[current_time_index]]
            )
        else:
            current_noise_floor_estimate_list = estimate_noise_floor(
                neural_data[
                    current_time_index - noise_floor_window : current_time_index
                ],
                window_size=noise_floor_window,
            )

        current_noise_floor_estimate = current_noise_floor_estimate_list[0]
        current_noise_floor_estimate_inverse = -(current_noise_floor_estimate)

        # Detect Initial First Point
        if initial_first_point_of_spike_detected == False:
            if current_time_index == 0:
                local_maximum_list_of_current_time_index = (
                    identify_potential_initial_spikes(
                        neural_data[current_time_index : current_time_index + 1]
                    )
                )
                is_current_time_index_local_maximum = (
                    local_maximum_list_of_current_time_index[0]
                )
            else:
                local_maximum_list_of_current_time_index = (
                    identify_potential_initial_spikes(
                        neural_data[current_time_index - 1 : current_time_index + 2]
                    )
                )
                is_current_time_index_local_maximum = (
                    local_maximum_list_of_current_time_index[1]
                )

            if is_current_time_index_local_maximum == True:
                # First Point Potentially Identified
                initial_first_point_of_spike_detected = True
                spike_time_index_first_point = current_time_index
        elif (
            second_point_of_spike_detected == False
            and initial_first_point_of_spike_detected == True
        ):
            # Detect Second Point
            local_minimum_list_of_current_time_index = (
                identify_potential_initial_spikes(
                    neural_data[current_time_index - 1 : current_time_index + 2],
                    return_local_maximum=False,
                )
            )
            is_current_time_index_local_minimum = (
                local_minimum_list_of_current_time_index[1]
            )
            if is_current_time_index_local_minimum == True:
                if (
                    neural_data[current_time_index]
                    < current_noise_floor_estimate_inverse
                ):
                    # Second Point Found
                    spike_time_index_list_first_to_second_points = np.arange(
                        start=spike_time_index_first_point,
                        stop=current_time_index,
                        step=1,
                    )
                    spike_time_index_second_point = current_time_index
                    second_point_of_spike_detected = True
                else:
                    initial_first_point_of_spike_detected = False
        elif (
            initial_first_point_of_spike_detected == True
            and second_point_of_spike_detected == True
            and third_point_of_spike_detected == False
        ):
            # Detect Third Point
            local_maximum_list_of_current_time_index = (
                identify_potential_initial_spikes(
                    neural_data[current_time_index - 1 : current_time_index + 2]
                )
            )
            is_current_time_index_local_maximum = (
                local_maximum_list_of_current_time_index[1]
            )
            if is_current_time_index_local_maximum == True:
                if neural_data[current_time_index] > current_noise_floor_estimate:
                    # Third Point Found
                    spike_time_index_list_second_to_third_points = np.arange(
                        spike_time_index_second_point, current_time_index + 1, step=1
                    )
                    third_point_of_spike_detected = True
                    time_index_of_most_recent_third_spike = current_time_index
                else:
                    initial_first_point_of_spike_detected = True
                    second_point_of_spike_detected = False
                    spike_time_index_first_point = current_time_index
        elif (
            initial_first_point_of_spike_detected == True
            and second_point_of_spike_detected == True
            and third_point_of_spike_detected == True
        ):
            # Detect Fourth Point
            if neural_data[current_time_index] < 0:
                time_index_of_most_recent_fourth_spike_point = current_time_index
                spike_time_index_list_third_to_fourth_points = np.arange(
                    time_index_of_most_recent_third_spike,
                    time_index_of_most_recent_fourth_spike_point + 1,
                    step=1,
                )
                spike_time_index_list = np.concatenate(
                    [
                        spike_time_index_list_first_to_second_points,
                        spike_time_index_list_second_to_third_points,
                        spike_time_index_list_third_to_fourth_points,
                    ]
                )
                spike_train_time_index_list.append(spike_time_index_list)

                initial_first_point_of_spike_detected = False
                second_point_of_spike_detected = False
                third_point_of_spike_detected = False
        else:
            raise ValueError("Error in Spike Detection State")

    return spike_train_time_index_list


def create_encoded_data(
    sample_rate,
    number_of_samples,
    spike_train_time_index_list,
    neural_data,
    time_array_of_neural_data,
):
    """This function creates an encoded version of the initial data.

    Args:
        spike_train_time_index_list (list): This is the list of array of
                                            floats that indicate indices
                                            of amplitudes.
        neural_data (array): These are the amplitudes of all values in
                             the dataset.
        time_array_of_neural_data (array): These are the points of time
                                           of the dataset.
        sample_rate (int): This is the sample rate of the data. The
                              samples are equidistant depending upon the
                              sampling frequency as calculated from the
                              inverse of the sample rate.
        number_of_samples (int): This is the total number of samples in
                                 the dataset.

    Returns:
        (list): This is the encoded data. This encoded data has the
                sample rate, the number of samples, the initial starting
                time of the first amplitude, and the information of the
                amplitudes of the detected eeg spikes. This pattern of
                the initial starting time of the first amplitude,
                represented as a float, followed by the array of
                amplitude values at each sample is repeated for each
                detected spike. It is implied that the samples are
                equidistant depending upon thesampling frequency as
                calculated from the inverse of thesample rate, that the
                length oftime of the entire datais inferred from the
                number of samplesdivided by thesample rate, and all
                amplitudes at samples notexplicitlydefined are to be
                considered noise and are therefore setto zero to
                reduce size while retaining information.
    """
    encoded_data = []
    encoded_data.append(sample_rate)
    encoded_data.append(number_of_samples)
    for spike_train_index in range(0, len(spike_train_time_index_list)):
        encoded_data.append(
            time_array_of_neural_data[spike_train_time_index_list[spike_train_index][0]]
        )
        encoded_data.append(neural_data[spike_train_time_index_list[spike_train_index]])
        encoded_data = deque(encoded_data)
    return encoded_data


def preprocess_signal(raw_neural_signal, sample_rate):
    """This function will process the raw neural signal by detrending
    then filtering the signal with a band-pass filter with a passband
    between 500 Hz and 5 KHz.

    Args:
        raw_neural_signal (ndarray): This is the array of amplitudes of
        a raw signal from the neuralink. This signal needs to be
        detrended and filtered to later extract the spike information
        contained within the signal.

    Returns:
        filtered_data_bandpass (ndarray): This is the array of the
        amplitude of the detrended, and band-pass filtered signal.
    """
    # Detrending the signal
    detrended_neural_data = scipy.signal.detrend(raw_neural_signal)

    # Normalize the signal between -1 and 1
    preprocessed_data = detrended_neural_data / detrended_neural_data.max()

    # Band-pass Filter
    nyq = sample_rate // 2
    low_cutoff_freq = 500
    high_cutoff_freq = 5000
    low = low_cutoff_freq / nyq
    high = high_cutoff_freq / nyq
    order = 4
    numerator, denominator = butter(order, [low, high], btype="band")

    filtered_data_bandpass = lfilter(numerator, denominator, preprocessed_data)
    return filtered_data_bandpass


def decode_data(encoded_data):
    """This function will decode the encoded file. It will convert the
    encoded format into an array of values containing only the
    amplitudes of the neural spike activity and zero-values in lieu of
    noise.

    Args:
        encoded_data (deque): This is the encoded data. It is a list
        where the first index is the sample rate, & the second index is
        the number of samples. The subsequent pair of indices contain
        the starting time of the first spike amplitude and the array of
        the amplitude values of the spike. This pattern follows for each
        detected spike in the original data.

    Returns:
        amplitude_array (ndarray): This is the array of spike amplitudes
                                   detected in the original signal. The
                                   noise of the signal has been
                                   nullified.
        sample_rate (int): This is the rate of the sample.
    """

    # Extract Metadata
    sample_rate = encoded_data.popleft()
    number_of_samples = encoded_data.popleft()

    # Construct the Time Array
    time_endpoint = number_of_samples / sample_rate
    time_array = np.arange(start=0, stop=time_endpoint, step=(1 / sample_rate))

    # Create the Amplitude Array
    amplitude_array = np.zeros(len(time_array))
    while len(encoded_data) > 0:
        amplitude_start_time = encoded_data.popleft()
        spike_amplitudes = encoded_data.popleft()
        for amplitude_index, amplitude in enumerate(spike_amplitudes):
            amplitude_array[
                np.where(time_array == amplitude_start_time)[0][0] + amplitude_index
            ] = amplitude
    return sample_rate, amplitude_array


def calculate_time_array(sample_rate: int, neural_data: np.ndarray):
    """This function creates the array of time values corresponding to
    the sample values in the raw_neural_data.

    Args:
        sample_rate (int): This is the rate the sample was taken.
        raw_neural_data (numpy.ndarray): This is the array of amplitudes.

    Returns:
        time_array_of_neural_data (numpy.ndarray): This is the array of
                                                   values where each
                                                   index corresponds to
                                                   the time width of the
                                                   frequency of the
                                                   sampling rate. The
                                                   frequency of the
                                                   sampling rate is
                                                   calculated as one
                                                   divided by the
                                                   sampling rate.
    """
    time_array_length = len(neural_data) / sample_rate
    time_array_of_neural_data = np.arange(
        start=0, stop=time_array_length, step=(1 / sample_rate)
    )
    return time_array_of_neural_data
