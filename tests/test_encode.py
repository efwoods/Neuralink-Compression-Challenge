"""This module is used to test the encode module."""

import unittest
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
import logging
from scipy.io import wavfile
import wave
import sys
import os
import numpy as np
import pickle

# Log all messages from all logging levels
logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from utility import signal_process

# Import encode
spec = spec_from_loader("encode", SourceFileLoader("encode", "./encode"))
encode = module_from_spec(spec)
spec.loader.exec_module(encode)


class TestEncode(unittest.TestCase):
    """This class is used to run test cases for the encode module.

    Args:
        unittest (module): This module allows unit tests to be run
        within the TestEncode class.
    """

    def setUp(self):
        self.file = "data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav"
        self.compressed_file_path = (
            "data/0ab237b7-fb12-4687-afed-8d1e2070d6" "21.wav.brainwire"
        )

    def tearDown(self):
        pass

    def test01_logging_and_test_methods(self):
        """Used to test the test methods and the logger
        functionality.
        """

        logging.info("test_logging_and_test_methods")
        print("test set up")
        logging.info("The logger works")

    def test02_read_input_wav_is_type_bytes(self):
        """Used to test the read_file method in the encode
        module.
        """

        logging.info("test_read_input_wav_is_type_bytes")
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        self.assertEqual(type(input_wav), np.ndarray)
        self.assertEqual(type(sample_rate), int)
        self.assertEqual(type(compressed_file_path), str)

    def test03_huffman_encoding_pickling(self):
        """This is a full test of the huffman encoding algorithm"""

        logging.info("Testing huffman encoding of pickled object format")
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        filtered_data_bandpass = signal_process.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        time_array_of_neural_data = signal_process.calculate_time_array(
            sample_rate=sample_rate, neural_data=filtered_data_bandpass
        )
        spike_train_time_index_list = signal_process.detect_neural_spikes(
            filtered_data_bandpass
        )
        encoded_data = signal_process.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
            time_array_of_neural_data=time_array_of_neural_data,
        )
        pickled_data = pickle.dumps(encoded_data)
        encode.huffman_encoding(
            pickled_data=pickled_data, compressed_file_path=self.compressed_file_path
        )

    def test04_read_wave_information(self):
        """This is a test that the information of the wave file is read."""

        input_wav = wave.open(self.file, "rb")
        logging.info("input_wav type: {}".format(type(input_wav)))
        logging.info("Channels: {}".format(input_wav.getnchannels()))
        logging.info("Sample width {} Bytes".format(input_wav.getsampwidth()))
        logging.info("Frequency: {}".format(input_wav.getframerate(), "kHz"))
        logging.info("Number of frames: {}".format(input_wav.getnframes()))
        logging.info(
            "Audio length: {:.2f} seconds".format(
                input_wav.getnframes() / input_wav.getframerate()
            )
        )
        pred_num_bytes = (
            input_wav.getnframes() * input_wav.getnchannels() * input_wav.getsampwidth()
        )

        sample_bytes = input_wav.readframes(input_wav.getnframes())
        self.assertEqual(pred_num_bytes, len(sample_bytes))
        logging.info(f"pred_num_bytes: {pred_num_bytes}")
        logging.info(f"len(sample_bytes): {len(sample_bytes)}")

    def test05_filter_modification_of_signal(self):
        """This is a test that the filters of the signal can be modified."""

        sample_rate, raw_signal_array = wavfile.read(self.file)
        fft, freq_bins = signal_process.preprocess_to_frequency_domain(
            raw_signal_array, sample_rate
        )
        percentage = 0.1

        filtered_fft = signal_process.modify_filters(fft, freq_bins, percentage)
        self.assertEqual(type(filtered_fft), np.ndarray)
        self.assertIsNotNone(filtered_fft)

    def test06_huffman_encoding_of_input_wav_file(self):
        """This is a test that the huffman encoding properly functions independently."""

        logging.info("Testing input of wavfile into huffman_encode function.")
        logging.info("Results: File Size: 128 KB in 0.078s")
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        encode.huffman_encoding(
            compressed_file_path=self.compressed_file_path, input_wave=input_wav
        )

    def test07_huffman_encoding_of_decoded_encoded_data(self):
        """This is a test to huffman encode data that contains only spike
        information where the noise has been zero-valued everywhere else."""

        logging.info("Testing using spike detection.")
        logging.info("Results: File Size: 132 KB in 3.962s")
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        filtered_data_bandpass = signal_process.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        time_array_of_neural_data = signal_process.calculate_time_array(
            sample_rate=sample_rate, neural_data=filtered_data_bandpass
        )
        spike_train_time_index_list = signal_process.detect_neural_spikes(
            filtered_data_bandpass
        )
        encoded_data = signal_process.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
            time_array_of_neural_data=time_array_of_neural_data,
        )
        sample_rate, amplitude_array = signal_process.decode_data(
            encoded_data=encoded_data
        )
        encode.huffman_encoding(
            compressed_file_path=self.compressed_file_path, input_wave=amplitude_array
        )

    def test08_writing_encoded_spikes_only(self):
        logging.info(
            "Testing File Size and Algorithmic Speed using the encoded information only"
        )
        logging.info("Results: File Size: 272 KB in 2.951s")
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        filtered_data_bandpass = signal_process.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        time_array_of_neural_data = signal_process.calculate_time_array(
            sample_rate=sample_rate, neural_data=filtered_data_bandpass
        )
        spike_train_time_index_list = signal_process.detect_neural_spikes(
            filtered_data_bandpass
        )
        encoded_data = signal_process.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
            time_array_of_neural_data=time_array_of_neural_data,
        )

        with open(compressed_file_path, "wb+") as file:
            file.write(pickle.dumps(encoded_data))
            file.close()


if __name__ == "__main__":
    unittest.main()
