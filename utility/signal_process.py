"""This module is used to modify filters & remove the noise of the
raw data."""

# Imports
import numpy as np
import scipy.signal

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
    threshold = percentage * (2 * np.abs(fft[0:len(fft)//2]) / len(freq_bins)).max()
    filtered_fft = fft.copy()
    filtered_fft_magnitude = np.abs(filtered_fft)
    filtered_fft_magnitude = 2 * filtered_fft_magnitude / len(freq_bins)
    filtered_fft[filtered_fft_magnitude <= threshold]=0
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
        freq_bins (np.ndarray): This is the bins of frequencies
                                pertaining to the fast fourier
                                transform.
    """
    detrended_signal = scipy.signal.detrend(raw_signal_array)
    fft = np.fft.fft(detrended_signal)
    freq_bins = np.arange(start=0, stop=(sample_rate // 2), 
                          step=(sample_rate / len(fft)))
    return fft, freq_bins
