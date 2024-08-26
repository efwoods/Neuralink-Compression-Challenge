"""This module is used to modify filters & remove the noise of the
raw data."""

# Imports
import numpy as np
import scipy.signal

from glob import glob
from scipy.signal import lfilter, butter
from collections import deque
import sys
import os
import time


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
    must be at least the length of the window size or a single value.

    Args:
        amplitude_array (numpy.ndarray): Array of amplitudes with which
                                         to derive the noise floor.

        window_size (int, optional): This is the width of the window
                                     used to calculate a rolling median
                                     average.

    Return:
        noise_floor_estimate (np.ndarray): This is the estimate of the
                                           noise floor.
    """
    if len(amplitude_array) == 0:
        raise ValueError("Length of amplitude array must be greater than 0")
    elif len(amplitude_array) == 1:
        noise_floor_estimate = np.array(
            [np.sqrt(np.abs(np.float64(amplitude_array[0])) ** 2)]
        )
        return noise_floor_estimate
    else:
        if len(amplitude_array) < window_size:
            window_size = len(amplitude_array)
        power_of_filtered_data = np.abs(np.float64(amplitude_array) ** 2)

        rolling_median_array = []
        for index in range(0, len(power_of_filtered_data), 1):
            current_median = np.median(
                power_of_filtered_data[index : index + window_size]
            )
            rolling_median_array.append(current_median)

        rolling_median_array = np.array(rolling_median_array)

        noise_floor_estimate = np.sqrt(rolling_median_array)

        return noise_floor_estimate


def detect_neural_spikes(neural_data, single_spike_detection=False):
    """This function detects a single neural spike in real-time and
    truncates the remaining neural data array if a flag is passed. This
    allows for a single spike to be compressed and sent as a file under
    time and file size constraints. It returns an array of indices of
    spike locations containing only an individual spike. This preserves
    the format for use with other functions but will inherently reduce
    the time to file and size of each file to meet the requirements of
    the problem statement. If the flag is not sent to the function, the
    the entire neural data is searched for spikes before the function is
    returned.

    Args:
        neural_data (array): This is the array of amplitudes for each
                             point of time of the neural data.
        single_spike_detection (bool): This is a boolean flag which will
                                       modify the function to detect
                                       single neural spikes and truncate
                                       the remaining neural data array.

    Returns:
        spike_train_time_index_list (list): This is the array inclusive
                                            of amplitudes of spikes at
                                            each specific point in the
                                            initial time array.
                                            Non-spike points have been
                                            replaced with amplitudes of
                                            zero value.
        neural_data (array): This is the array of neural data. If
                             single_spike_detection is set to True, then
                             the neural data array has been truncated to
                             remove values up to the final point
                             detected on the spike.
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

                if single_spike_detection == True:
                    neural_data = neural_data[current_time_index:]
                    break
                else:
                    initial_first_point_of_spike_detected = False
                    second_point_of_spike_detected = False
                    third_point_of_spike_detected = False
        else:
            raise ValueError("Error in Spike Detection State")

    return spike_train_time_index_list, neural_data


def create_encoded_data(
    sample_rate,
    number_of_samples,
    spike_train_time_index_list,
    neural_data,
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
        encoded_data (list): This is the encoded data. This encoded data
                             has the sample rate, the number of samples,
                             the initial start time index of the first
                             amplitude, and the information of the
                             amplitudes of the detected eeg spikes.
                             This pattern of the initial time index
                             of the first amplitude, represented as an
                             int, followed by the number of points in
                             the detected spike, followed by the array
                             of amplitude values at each sample is
                             repeated for each detected spike. It is
                             implied that the samples are equidistant
                             depending upon the sampling frequency as
                             calculated from the inverse of the sample
                             rate, that the length of time of the entire
                             data is inferred from the number of samples
                             divided by the sample rate, and all
                             amplitudes at samples not explicitly
                             defined are to be considered noise and are
                             therefore set to zero to reduce size while
                             retaining information. The time of each
                             amplitude is calculated as the division of
                             the starting time index plus the current
                             position of each amplitude by the current
                             position in the zero-based amplitude array
                             by the sample rate.
    """
    encoded_data = []
    encoded_data.append(np.int32(sample_rate))
    encoded_data.append(np.int32(number_of_samples))
    for spike_train_index, spike_train_value in enumerate(spike_train_time_index_list):
        # Time index of the first spike point
        encoded_data.append(np.int32(spike_train_value[0]))
        # The number of points in the detected spike to decode the byte string.
        encoded_data.append(np.int32(len(neural_data[spike_train_value])))
        # The amplitude array of points in the spike.
        encoded_data.append(neural_data[spike_train_value])

    return encoded_data


def preprocess_signal(raw_neural_signal, sample_rate):
    """This function will process the raw neural signal by detrending
    then filtering the signal with a band-pass filter with a passband
    between 500 Hz and 5 KHz.

    Args:
        raw_neural_signal (ndarray): This is the array of amplitudes of
                                     a raw signal from the neuralink.
                                     This signal needs to be detrended
                                     and filtered to later extract the
                                     spike information contained within
                                     the signal.

    Returns:
        filtered_data_bandpass (ndarray): This is the array of the
                                          amplitude of the detrended,
                                          and band-pass filtered signal.
    """
    # Detrending the signal
    detrended_neural_data = np.int16(scipy.signal.detrend(raw_neural_signal))

    # Band-pass Filter
    nyq = sample_rate // 2
    low_cutoff_freq = 500
    high_cutoff_freq = 5000
    low = low_cutoff_freq / nyq
    high = high_cutoff_freq / nyq
    order = 4
    numerator, denominator = butter(order, [low, high], btype="band")

    filtered_data_bandpass = np.int16(
        lfilter(numerator, denominator, detrended_neural_data)
    )
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
    encoded_data = deque(encoded_data)
    sample_rate = encoded_data.popleft()
    number_of_samples = encoded_data.popleft()

    # Construct the Time Array
    time_endpoint = number_of_samples / sample_rate
    time_array = np.arange(start=0, stop=time_endpoint, step=(1 / sample_rate))

    # Create the Amplitude Array
    amplitude_array = np.int16(np.zeros(len(time_array)))
    while len(encoded_data) > 0:
        amplitude_start_time_index = encoded_data.popleft()
        number_of_spike_points = encoded_data.popleft()
        spike_amplitudes = encoded_data.popleft()
        for amplitude_index, amplitude in enumerate(spike_amplitudes):
            amplitude_array[amplitude_start_time_index + amplitude_index] = amplitude
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


def convert_encoded_data_to_byte_string(encoded_data: list):
    """This converts the encoded data to the a string of bytes.

    Args:
        encoded_data (list): This is the array of values to be converted
                             into a string of bytes.

    Returns:
        encoded_data_byte_string (str): This is the string of bytes that
                                        represent the encoded data.
    """
    byte_string = encoded_data[0].tobytes()
    for data in encoded_data[1:]:
        byte_string += data.tobytes()

    return byte_string


def convert_byte_string_to_encoded_data(encoded_data_byte_string: str):
    """This converts the string of bytes to the encoded data
    representation of the input wav so that it may be decoded and
    assembled into an array of amplitudes.

    Args:
        encoded_data_byte_string (str): This is the string of bytes that
                                        represent the encoded data.
    Returns:
        encoded_data (list): This is the list of integers which contain
                             the spike information of the dissassembled
                             amplitude array.
    """
    encoded_data = []

    # Sample Rate:
    encoded_data.append(np.frombuffer(encoded_data_byte_string[0:4], dtype=np.int32)[0])
    encoded_data_byte_string = encoded_data_byte_string[4:]

    # Number of Samples:
    encoded_data.append(np.frombuffer(encoded_data_byte_string[0:4], dtype=np.int32)[0])
    encoded_data_byte_string = encoded_data_byte_string[4:]

    while len(encoded_data_byte_string) > 0:
        # Time index of first spike point:
        encoded_data.append(
            np.frombuffer(encoded_data_byte_string[0:4], dtype=np.int32)[0]
        )
        encoded_data_byte_string = encoded_data_byte_string[4:]

        # Number of points in the following spike amplitude array:
        number_of_points_in_the_spike_amplitude_array = np.frombuffer(
            encoded_data_byte_string[0:4], dtype=np.int32
        )[0]
        encoded_data.append(number_of_points_in_the_spike_amplitude_array)
        encoded_data_byte_string = encoded_data_byte_string[4:]

        # Array of spike amplitudes:
        encoded_data.append(
            np.frombuffer(
                encoded_data_byte_string[
                    0 : 2 * number_of_points_in_the_spike_amplitude_array
                ],
                dtype=np.int16,
            )
        )
        encoded_data_byte_string = encoded_data_byte_string[
            2 * number_of_points_in_the_spike_amplitude_array :
        ]

    return encoded_data


def print_size_of_file_compression(file_path: str, compressed_file_path: str):
    """This function prints the file size, the compressed file size, and
    the percent the file has been compressed.

    Args:
        file_path (str): This is the path of the original file.
        compressed_file_path (str): This is the path of the compressed file.
    """
    file_size = os.path.getsize(file_path)
    compressed_file_size = os.path.getsize(compressed_file_path)
    percent_of_compression = (1 - (compressed_file_size / file_size)) * 100
    file_size_requirement = file_size // 200
    percent_of_file_size_relative_to_file_size_requirement = (
        compressed_file_size / file_size_requirement
    ) * 100
    print(f"Original File Size: {file_size}")
    print(f"Compressed File Size: {compressed_file_size}")
    print(f"Percent of Compression: {percent_of_compression:.2f}%")
    print(
        f"Percent of Compressed File Size Relative to Required File Size:{percent_of_file_size_relative_to_file_size_requirement:.3f}%"
    )


def print_time_each_function_takes_to_complete_processing(
    start_time: int, stop_time: int, executed_line: str = None
):
    """This function prints the time delta between the start time and the stop time.

    Args:
        start_time (int): This is the integer representation of the start time in nanoseconds.
        stop_time (int): This is the integer representation of teh stop time in nanoseconds.
        executed_line (str, optional): This is the line of code that was executed. Defaults to None.
    """
    time_Δ = stop_time - start_time
    if executed_line != None:
        executed_line_str = "Executed Line: "
        executed_line_str += executed_line
        executed_line_str += "..."
        print(f"\n{executed_line_str}")
    else:
        print(f"\n")
    print(f"Time Δ Nanoseconds: {(time_Δ)}")
    print(f"Time Δ Microseconds: {(time_Δ / 1e3)}")
    print(f"Time Δ Milliseconds: {(time_Δ / 1e6)}")
    print(f"Time Δ Seconds: {(time_Δ / 1e9)}")
    print(f"\n")
