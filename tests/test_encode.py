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
import time

# Log all messages from all logging levels
logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from utility import signal_process

# Import encode
spec = spec_from_loader("encode", SourceFileLoader("encode", "./encode"))
encode = module_from_spec(spec)
spec.loader.exec_module(encode)

# Import decode
spec = spec_from_loader("decode", SourceFileLoader("decode", "./decode"))
decode = module_from_spec(spec)
spec.loader.exec_module(decode)


class TestEncode(unittest.TestCase):
    """This class is used to run test cases for the encode module.

    Args:
        unittest (module): This module allows unit tests to be run
        within the TestEncode class.
    """

    def setUp(self):
        self.file = "data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav"
        self.compressed_file_path = (
            "data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav.brainwire"
        )

    def tearDown(self):
        pass

    @unittest.skip("Testing Only Highest Compression Ratio")
    def test01_logging_and_test_methods(self):
        """Used to test the test methods and the logger
        functionality.
        """

        logging.info("test_logging_and_test_methods")
        print("test set up")
        logging.info("The logger works")

    @unittest.skip("Testing Only Highest Compression Ratio")
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

    @unittest.skip("Testing Only Highest Compression Ratio")
    def test03_huffman_encoding_pickling(self):
        """This is a full test of the huffman encoding algorithm"""

        logging.info("Testing huffman encoding of pickled object format")
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        filtered_data_bandpass = signal_process.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list, neural_data = signal_process.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = signal_process.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )
        pickled_data = pickle.dumps(encoded_data)
        encode.huffman_encoding(
            pickled_data=pickled_data, compressed_file_path=self.compressed_file_path
        )

    @unittest.skip("Testing Only Highest Compression Ratio")
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

    @unittest.skip("Testing Only Highest Compression Ratio")
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

    @unittest.skip("Testing Only Highest Compression Ratio")
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

    @unittest.skip("Testing Only Highest Compression Ratio")
    def test07_huffman_encoding_of_decoded_encoded_data(self):
        """This is a test to huffman encode data that contains only spike
        information where the noise has been zero-valued everywhere else."""

        logging.info("Testing using spike detection.")
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        filtered_data_bandpass = signal_process.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list, neural_data = signal_process.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = signal_process.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )
        sample_rate, amplitude_array = signal_process.decode_data(
            encoded_data=encoded_data
        )
        encode.huffman_encoding(
            compressed_file_path=self.compressed_file_path, input_wave=amplitude_array
        )

    @unittest.skip("Testing Only Highest Compression Ratio")
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
        spike_train_time_index_list, neural_data = signal_process.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = signal_process.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )

        with open(compressed_file_path, "wb+") as file:
            file.write(pickle.dumps(encoded_data))
            file.close()

    def test09_writing_encoded_data_byte_string_using_huffman_encoding(self):
        logging.info(
            "Testing Using Huffman Encoding on the String of Bytes that Contain Only Detected Spike Information."
        )
        total_start_time = time.time_ns()
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        filtered_data_bandpass = signal_process.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list, neural_data = signal_process.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = signal_process.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )
        encoded_data_byte_string = signal_process.convert_encoded_data_to_byte_string(
            encoded_data
        )

        encode.huffman_encoding(
            input_data=encoded_data_byte_string,
            compressed_file_path=self.compressed_file_path,
        )
        total_stop_time = time.time_ns()

        signal_process.print_size_of_file_compression(
            file_path=self.file,
            compressed_file_path=self.compressed_file_path,
        )

        signal_process.print_time_each_function_takes_to_complete_processing(
            start_time=total_start_time, stop_time=total_stop_time
        )

    @unittest.skip("skipping test")
    def test10_detect_single_neural_spikes(self):
        logging.info("This function tests the ability to detect single neural spikes.")
        total_start_time = time.time_ns()
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        filtered_data_bandpass = signal_process.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )

        # Spike Train Time Index List Should Contain A Single Spike
        spike_train_time_index_list, truncated_neural_data = (
            signal_process.detect_neural_spikes(
                neural_data=filtered_data_bandpass, single_spike_detection=True
            )
        )
        encoded_data = signal_process.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=truncated_neural_data,
        )
        encoded_data_byte_string = signal_process.convert_encoded_data_to_byte_string(
            encoded_data
        )
        encode.huffman_encoding(
            input_data=encoded_data_byte_string,
            compressed_file_path=self.compressed_file_path,
        )
        total_stop_time = time.time_ns()
        signal_process.print_time_each_function_takes_to_complete_processing(
            start_time=total_start_time, stop_time=total_stop_time
        )
        signal_process.print_size_of_file_compression(
            file_path=self.file,
            compressed_file_path=self.compressed_file_path,
        )

    @unittest.skip("Testing Encoding & Decoding Functionality")
    def test11_writing_encoded_data_byte_string_(self):
        logging.info(
            "Testing Efficiency of Writing String of Bytes that Contain Only Detected Spike Information."
        )
        total_start_time = time.time_ns()
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        filtered_data_bandpass = signal_process.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list, neural_data = signal_process.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = signal_process.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )
        encoded_data_byte_string = signal_process.convert_encoded_data_to_byte_string(
            encoded_data
        )
        with open(self.compressed_file_path, "wb+") as fp:
            fp.write(encoded_data_byte_string)
            fp.close()
        total_stop_time = time.time_ns()
        signal_process.print_time_each_function_takes_to_complete_processing(
            start_time=total_start_time, stop_time=total_stop_time
        )
        signal_process.print_size_of_file_compression(
            file_path=self.file,
            compressed_file_path=self.compressed_file_path,
        )

    @unittest.skip("Testing Compression Highest Compression Ratio")
    def test12_testing_writing_input_wav(self):
        logging.info(
            "This is a test to ensure the input wav is written to the output file and is functional."
        )
        decompressed_file_path = "data/_0ab237b7-fb12-4687-afed-8d1e2070d621.wav"

        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )

        wavfile.write(filename=decompressed_file_path, rate=sample_rate, data=input_wav)

    def test13_writing_encoded_data_byte_string_using_huffman_encoding(self):
        logging.info(
            "Testing Using Huffman Encoding on the String of Bytes that Contain Only Detected Spike Information."
        )
        file = "data/0052503c-2849-4f41-ab51-db382103690c.wav"
        compressed_file_path = "data/0052503c-2849-4f41-ab51-db382103690c.wav.brainwire"

        total_start_time = time.time_ns()
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            file, compressed_file_path
        )
        filtered_data_bandpass = signal_process.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list, neural_data = signal_process.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = signal_process.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )
        encoded_data_byte_string = signal_process.convert_encoded_data_to_byte_string(
            encoded_data
        )

        encode.huffman_encoding(
            input_data=encoded_data_byte_string,
            compressed_file_path=self.compressed_file_path,
        )
        total_stop_time = time.time_ns()

        signal_process.print_size_of_file_compression(
            file_path=self.file,
            compressed_file_path=self.compressed_file_path,
        )

        signal_process.print_time_each_function_takes_to_complete_processing(
            start_time=total_start_time, stop_time=total_stop_time
        )


if __name__ == "__main__":
    unittest.main()
