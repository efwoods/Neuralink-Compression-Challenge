"""This module is used to modify filters & remove the noise of the
raw data."""

# Imports
import numpy as np

def main(fft, freq_bins, percentage):
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
