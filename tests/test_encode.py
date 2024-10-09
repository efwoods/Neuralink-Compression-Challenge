"""This module is used to test the encode module."""

import unittest
import logging
from scipy.io import wavfile
import wave
import numpy as np
import pickle
import time
from signal_processing_utilities import process_signal
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader

# Importing local version of "encode" file
spec = spec_from_loader("encode", SourceFileLoader("encode", "./encode"))
encode = module_from_spec(spec)
spec.loader.exec_module(encode)

# Log all messages from all logging levels
logging.basicConfig(level=logging.DEBUG)


# Helper Functions
def print_differences_in_array(x_df, verbose=False):
    """This function will print differences between values in a given
       array.

    Args:
        x_df (numpy.ndarray): This is the numpy array of which to print
                              differences.
    """
    count = 0
    duplicate_list = []
    for index in range(0, len(x_df)):
        if index == 0:
            continue
        difference = x_df[index] - x_df[index - 1]
        if difference == 0:
            if verbose:
                print(f"difference: {difference}\nindex: {index}")
            duplicate_list.append(index)
            count += 1
    return count, duplicate_list


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
        self.debug_file = "data/0052503c-2849-4f41-ab51-db382103690c.wav"
        self.debug_compressed_file_path = (
            "data/0052503c-2849-4f41-ab51-db382103690c.wav.brainwire"
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

        sample_rate, input_wav = wavfile.read(filename=self.file)
        logging.info("test_read_input_wav_is_type_bytes")

        self.assertEqual(type(input_wav), np.ndarray)
        self.assertEqual(type(sample_rate), int)

    def test03_huffman_encoding_pickling(self):
        """Testing Reading Data, Filtering the Data, Detecting Neural
        Spikes, & Creating Encoded Data"""

        logging.info("Testing huffman encoding of pickled object format")
        sample_rate, input_wav = wavfile.read(filename=self.file)

        filtered_data_bandpass = process_signal.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list = process_signal.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = process_signal.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
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

        sample_rate, raw_signal_array = wavfile.read(filename=self.file)
        fft, freq_bins = process_signal.preprocess_to_frequency_domain(
            raw_signal_array, sample_rate
        )
        percentage = 0.1

        filtered_fft = process_signal.modify_filters(fft, freq_bins, percentage)
        self.assertEqual(type(filtered_fft), np.ndarray)
        self.assertIsNotNone(filtered_fft)

    def test06_huffman_encoding_of_input_wav_file(self):
        """This is a test that the huffman encoding properly functions independently."""

        logging.info("Testing input of wavfile into huffman_encode function.")
        logging.info(
            "The sample rate is implied using this method to be a known value of 19531."
        )
        sample_rate, input_wav = wavfile.read(filename=self.file)

        node_mapping_dict, bit_string, end_zero_padding = encode.huffman_encoding(
            input_data=input_wav
        )

        byte_string = encode.create_byte_string(
            node_mapping_dict, bit_string, end_zero_padding, method_of_compression="h"
        )

        process_signal.write_file_bytes(
            file_path=self.compressed_file_path, data_bytes=byte_string
        )

        process_signal.print_size_of_file_compression(
            file_path=self.file, compressed_file_path=self.compressed_file_path
        )

    def test07_huffman_encoding_of_decoded_encoded_data(self):
        """This is a test to huffman encode data that contains only spike
        information where the noise has been zero-valued everywhere else."""

        logging.info("Testing using spike detection.")
        sample_rate, input_wav = wavfile.read(filename=self.file)
        filtered_data_bandpass = process_signal.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list = process_signal.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = process_signal.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )
        sample_rate, amplitude_array = process_signal.decode_data(
            encoded_data=encoded_data
        )
        node_mapping_dict, bit_string, end_zero_padding = encode.huffman_encoding(
            input_data=amplitude_array
        )

        # This is a quick method of compression because only huffman
        # encoding is truly implemented.
        byte_string = encode.create_byte_string(
            node_mapping_dict, bit_string, end_zero_padding, method_of_compression="h"
        )

        process_signal.write_file_bytes(
            file_path=self.compressed_file_path, data_bytes=byte_string
        )

    def test08_writing_encoded_spikes_only(self):
        logging.info(
            "Testing File Size and Algorithmic Speed using "
            + "the encoded information only"
        )
        sample_rate, input_wav = wavfile.read(filename=self.file)
        filtered_data_bandpass = process_signal.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list = process_signal.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = process_signal.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )

        with open(self.compressed_file_path, "wb+") as file:
            file.write(pickle.dumps(encoded_data))
            file.close()

        pickle_encoded_data_file_size = process_signal.print_file_size(
            file_path=self.compressed_file_path
        )

    def test09_writing_encoded_data_byte_string_using_huffman_encoding_main(self):
        logging.info(
            "This is the Main Function: Testing Using Huffman "
            + "Encoding on the String of Bytes that Contain Only "
            + "Detected Spike Information."
        )
        total_start_time = time.time_ns()
        start_time = time.time_ns()
        sample_rate, input_wav = wavfile.read(filename=self.file)
        stop_time = time.time_ns()
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=start_time,
            stop_time=stop_time,
            executed_line="wavfile.read(",
        )

        start_time = time.time_ns()
        filtered_data_bandpass = process_signal.preprocess_signal(
            raw_neural_signal=input_wav,
            sample_rate=sample_rate,
        )
        stop_time = time.time_ns()
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=start_time,
            stop_time=stop_time,
            executed_line="process_signal.preprocess_signal(",
        )

        start_time = time.time_ns()
        spike_train_time_index_list = process_signal.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        stop_time = time.time_ns()
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=start_time,
            stop_time=stop_time,
            executed_line="process_signal.detect_neural_spikes(",
        )

        start_time = time.time_ns()
        encoded_data = process_signal.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )
        stop_time = time.time_ns()
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=start_time,
            stop_time=stop_time,
            executed_line="process_signal.create_encoded_data(",
        )

        start_time = time.time_ns()
        encoded_data_byte_string = process_signal.convert_encoded_data_to_byte_string(
            encoded_data
        )
        stop_time = time.time_ns()
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=start_time,
            stop_time=stop_time,
            executed_line="process_signal.convert_encoded_data_to_byte_string(",
        )

        start_time = time.time_ns()
        node_mapping_dict, bit_string, end_zero_padding = encode.huffman_encoding(
            input_data=encoded_data_byte_string,
        )

        byte_string = encode.create_byte_string(
            node_mapping_dict, bit_string, end_zero_padding, method_of_compression="n"
        )
        stop_time = time.time_ns()
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=start_time,
            stop_time=stop_time,
            executed_line="encode.huffman_encoding(",
        )

        start_time = time.time_ns()
        process_signal.write_file_bytes(
            file_path=self.compressed_file_path, data_bytes=byte_string
        )
        stop_time = time.time_ns()

        total_stop_time = time.time_ns()

        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=start_time,
            stop_time=stop_time,
            executed_line="process_signal.write_file_bytes(",
        )

        process_signal.print_size_of_file_compression(
            file_path=self.file,
            compressed_file_path=self.compressed_file_path,
        )

        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=total_start_time,
            stop_time=total_stop_time,
            executed_line="Total Compression Time",
        )
        huffman_encoded_data_file_size = process_signal.print_file_size(
            file_path=self.compressed_file_path
        )

    def test10_detect_single_neural_spikes(self):
        logging.info("This function tests the ability to detect single neural spikes.")
        total_start_time = time.time_ns()
        sample_rate, input_wav = wavfile.read(filename=self.file)
        filtered_data_bandpass = process_signal.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )

        spike_train_time_index_list = process_signal.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=True
        )

        # Spike Train Time Index List Should Contain A Single Spike
        self.assertEqual(len(spike_train_time_index_list), 1)

        encoded_data = process_signal.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )
        encoded_data_byte_string = process_signal.convert_encoded_data_to_byte_string(
            encoded_data
        )
        node_mapping_dict, bit_string, end_zero_padding = encode.huffman_encoding(
            input_data=encoded_data_byte_string,
        )

        byte_string = encode.create_byte_string(
            node_mapping_dict, bit_string, end_zero_padding, method_of_compression="n"
        )

        process_signal.write_file_bytes(
            file_path=self.compressed_file_path, data_bytes=byte_string
        )

        total_stop_time = time.time_ns()

        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=total_start_time, stop_time=total_stop_time
        )
        process_signal.print_size_of_file_compression(
            file_path=self.file,
            compressed_file_path=self.compressed_file_path,
        )

    def test11_writing_encoded_data_byte_string_(self):
        logging.info(
            "Testing Efficiency of Writing String of Bytes that "
            + "Contain Only Detected Spike Information."
        )
        total_start_time = time.time_ns()
        sample_rate, input_wav = wavfile.read(filename=self.file)
        filtered_data_bandpass = process_signal.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list = process_signal.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = process_signal.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )
        encoded_data_byte_string = process_signal.convert_encoded_data_to_byte_string(
            encoded_data
        )
        with open(self.compressed_file_path, "wb+") as fp:
            fp.write(encoded_data_byte_string)
            fp.close()
        total_stop_time = time.time_ns()
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=total_start_time, stop_time=total_stop_time
        )
        process_signal.print_size_of_file_compression(
            file_path=self.file,
            compressed_file_path=self.compressed_file_path,
        )

    def test12_testing_writing_input_wav(self):
        logging.info(
            "This is a test to ensure the input wav is written to "
            + "the output file and is functional."
        )

        sample_rate, input_wav = wavfile.read(
            filename=self.debug_file,
        )

        wavfile.write(
            filename=self.debug_compressed_file_path, rate=sample_rate, data=input_wav
        )

    def test13_writing_encoded_data_byte_string_using_huffman_encoding(self):
        logging.info(
            "Testing Using Huffman Encoding on the String of Bytes "
            + "that Contain Only Detected Spike Information."
        )

        total_start_time = time.time_ns()
        sample_rate, input_wav = wavfile.read(filename=self.debug_file)
        filtered_data_bandpass = process_signal.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list = process_signal.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )
        encoded_data = process_signal.create_encoded_data(
            sample_rate=sample_rate,
            number_of_samples=len(filtered_data_bandpass),
            spike_train_time_index_list=spike_train_time_index_list,
            neural_data=filtered_data_bandpass,
        )
        encoded_data_byte_string = process_signal.convert_encoded_data_to_byte_string(
            encoded_data
        )

        node_mapping_dict, bit_string, end_zero_padding = encode.huffman_encoding(
            input_data=encoded_data_byte_string,
        )

        byte_string = encode.create_byte_string(
            node_mapping_dict, bit_string, end_zero_padding, method_of_compression="n"
        )

        process_signal.write_file_bytes(
            file_path=self.debug_compressed_file_path, data_bytes=byte_string
        )

        total_stop_time = time.time_ns()

        process_signal.print_size_of_file_compression(
            file_path=self.file,
            compressed_file_path=self.debug_compressed_file_path,
        )

        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=total_start_time, stop_time=total_stop_time
        )

    def test14_test_that_duplicates_do_not_exist_in_the_spike_train_time_index_list(
        self,
    ):
        logging.info(
            "Testing that there are no duplicates in the "
            + "spike_train_time_index_list"
        )
        sample_rate, input_wav = wavfile.read(filename=self.file)

        filtered_data_bandpass = process_signal.preprocess_signal(
            raw_neural_signal=input_wav, sample_rate=sample_rate
        )
        spike_train_time_index_list = process_signal.detect_neural_spikes(
            neural_data=filtered_data_bandpass, single_spike_detection=False
        )

        # Test to Detect Duplicates
        count = 0
        duplicate_list = []
        for index in range(0, len(spike_train_time_index_list)):
            current_count, current_duplicate_list = print_differences_in_array(
                spike_train_time_index_list[index]
            )
            count += current_count
            duplicate_list.append(current_duplicate_list)
        duplicate_list = np.array(duplicate_list)
        # If duplicate_list.shape[1] is 0, there are no duplicates:
        self.assertEqual(duplicate_list.shape[1], 0)

    def test15_test_of_arg_parser(self):
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-q"])

        # Asserting args for quick compression
        self.assertEqual(args.quick, True)
        self.assertEqual(args.file_path, self.file)
        self.assertEqual(args.compressed_file_path, self.compressed_file_path)

        # Asserting args for spike detection module
        args = parser.parse_args([self.file, self.compressed_file_path])
        self.assertEqual(args.quick, False)
        self.assertEqual(args.file_path, self.file)
        self.assertEqual(args.compressed_file_path, self.compressed_file_path)

    def test16_test_compress_file_name(self):
        logging.info(
            "This is a test to compress the data using the "
            + "'compress' method and the file name."
        )
        byte_string = encode.compress(file=self.file)
        self.assertEqual(type(byte_string), bytes)

    def test17_test_compress_file_name_quick(self):
        logging.info(
            "This is a test to compress the data using the "
            + "'compress' method and the file name where "
            + "the quick argument is passed into the function."
        )
        byte_string = encode.compress(file=self.file, quick=True)
        self.assertEqual(type(byte_string), bytes)

    def test18_test_compress_sample_rate_input_wav(self):
        logging.info(
            "This is a test to compress the data using the "
            + "'compress' method where the inputs are "
            + "sample_rate and the input_wav."
        )
        sample_rate, input_wav = wavfile.read(filename=self.file)
        byte_string = encode.compress(sample_rate=sample_rate, input_wav=input_wav)
        self.assertEqual(type(byte_string), bytes)

    def test19_test_compress_sample_rate_input_wav_quick(self):
        logging.info(
            "This is a test to compress the data using the "
            + "'compress' method where the inputs are "
            + "sample_rate and input_wav while implementing the "
            + "'quick' option. "
        )
        sample_rate, input_wav = wavfile.read(filename=self.file)
        byte_string = encode.compress(
            sample_rate=sample_rate, input_wav=input_wav, quick=True
        )
        self.assertEqual(type(byte_string), bytes)

    def test20_test_main_method_of_compression_q(self):
        logging.info(
            "This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'h' which indicates a huffman encoding "
            + "format exclusively."
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=h"])
        encode.main(args)

    def test21_test_main_method_of_compression_u(self):
        logging.info(
            "This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'u' which indicates implementing huffman encoding "
            + "and a unique amplitudes list."
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=u"])
        encode.main(args)

    def test22_test_main_method_of_compression_n(self):
        logging.info(
            "This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'n' which indicates implementing neural spike "
            + "detection."
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=n"])
        encode.main(args)


if __name__ == "__main__":

    unittest.main()
