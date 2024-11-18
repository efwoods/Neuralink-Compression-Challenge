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
import heapq

# Importing local version of "encode" file
spec = spec_from_loader("encode", SourceFileLoader("encode", "./encode"))
encode = module_from_spec(spec)
spec.loader.exec_module(encode)

# Custom import of local file "decode"
spec = spec_from_loader("decode", SourceFileLoader("decode", "./decode"))
decode = module_from_spec(spec)
spec.loader.exec_module(decode)

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
        self.test_sample_rate = test_sample_rate = 19531
        self.test_data = test_data = np.arange(0, 100000, step=1, dtype=np.int32)
        
        self.test_file_path = 'test_file.wav'
        self.test_compressed_file_path = 'test_file.wav.brainwire'
        

    def tearDown(self):
        pass

    def test01_logging_and_test_methods(self):
        """Used to test the test methods and the logger
        functionality.
        """

        logging.info("\n\ntest01: test_logging_and_test_methods\n\n")
        print("test set up")
        logging.info("The logger works")

    def test02_print_differences_in_array(self):
        logging.info(
            "\n\ntest02: This test ensures the helper function "
            + "'print_differences_in_array is operational.\n\n"
        )
        test_df = np.arange(0, 100, step=1)

        count, duplicate_list = print_differences_in_array(test_df, verbose=True)

        self.assertTrue(type(count), int)
        self.assertTrue(type(duplicate_list), list)

    def test03_read_input_wav_is_type_bytes(self):
        """Used to test the read_file method in the encode
        module.
        """

        sample_rate, input_wav = wavfile.read(filename=self.file)
        logging.info("\n\ntest03: test_read_input_wav_is_type_bytes\n\n")

        self.assertEqual(type(input_wav), np.ndarray)
        self.assertEqual(type(sample_rate), int)

    def test04_huffman_encoding_pickling(self):
        """Testing Reading Data, Filtering the Data, Detecting Neural
        Spikes, & Creating Encoded Data"""

        logging.info(
            "\n\ntest04: Testing huffman encoding of pickled object format\n\n"
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

    def test05_read_wave_information(self):
        """This is a test that the information of the wave file is read."""

        logging.info(
            "\n\ntest05: This is a test that the information of the wave file is read. \n\n"
        )
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

    def test06_filter_modification_of_signal(self):
        """This is a test that the filters of the signal can be modified."""

        logging.info(
            "\n\nThis is a test that the filters of the signal can be modified.\n\n"
        )
        sample_rate, raw_signal_array = wavfile.read(filename=self.file)
        fft, freq_bins = process_signal.preprocess_to_frequency_domain(
            raw_signal_array, sample_rate
        )
        percentage = 0.1

        filtered_fft = process_signal.modify_filters(fft, freq_bins, percentage)
        self.assertEqual(type(filtered_fft), np.ndarray)
        self.assertIsNotNone(filtered_fft)

    def test07_huffman_encoding_of_input_wav_file(self):
        """This is a test that the huffman encoding properly functions independently."""

        logging.info(
            "\n\ntest07: Testing input of wavfile into huffman_encode function.\n\n"
        )
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

    def test08_huffman_encoding_of_decoded_encoded_data(self):
        """This is a test to huffman encode data that contains only spike
        information where the noise has been zero-valued everywhere else."""

        logging.info("\n\ntest08: Testing using spike detection.\n\n")
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

    def test09_writing_encoded_spikes_only(self):
        logging.info(
            "\n\ntest09: Testing File Size and Algorithmic Speed using "
            + "the encoded information only. \n\n"
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

    def test10_writing_encoded_data_byte_string_using_huffman_encoding_main(self):
        logging.info(
            "\n\n test10: This is the Main Function: Testing Using Huffman "
            + "Encoding on the String of Bytes that Contain Only "
            + "Detected Spike Information.\n\n"
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

    def test11_detect_single_neural_spikes(self):
        logging.info(
            "\n\ntest11: This function tests the ability to detect single neural spikes.\n\n"
        )
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

    def test12_writing_encoded_data_byte_string_(self):
        logging.info(
            "\n\n test12: Testing Efficiency of Writing String of Bytes that "
            + "Contain Only Detected Spike Information.\n\n"
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

    def test13_testing_writing_input_wav(self):
        logging.info(
            "\n\n test13: This is a test to ensure the input wav is written to "
            + "the output file and is functional.\n\n"
        )

        sample_rate, input_wav = wavfile.read(
            filename=self.debug_file,
        )

        wavfile.write(
            filename=self.debug_compressed_file_path, rate=sample_rate, data=input_wav
        )

    def test14_writing_encoded_data_byte_string_using_huffman_encoding(self):
        logging.info(
            "\n\ntest14: Testing Using Huffman Encoding on the String of Bytes "
            + "that Contain Only Detected Spike Information.\n\n"
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

    def test15_test_that_duplicates_do_not_exist_in_the_spike_train_time_index_list(
        self,
    ):
        logging.info(
            "\n\ntest15: Testing that there are no duplicates in the "
            + "spike_train_time_index_list. \n\n"
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

    def test16_test_of_arg_parser(self):
        logging.info(
            "\n\ntest16: This is a test to ensure the argument parser "
            + "successfully parses the arguments.\n\n"
        )
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

    def test17_test_compress_file_name(self):
        logging.info(
            "\n\ntest17: This is a test to compress the data using the "
            + "'compress' method and the file name.\n\n"
        )
        byte_string = encode.compress(file=self.file)
        self.assertEqual(type(byte_string), bytes)

    def test18_test_compress_file_name_quick(self):
        logging.info(
            "\n\ntest18: This is a test to compress the data using the "
            + "'compress' method and the file name where "
            + "the quick argument is passed into the function.\n\n"
        )
        byte_string = encode.compress(file=self.file, quick=True)
        self.assertEqual(type(byte_string), bytes)

    def test19_test_compress_sample_rate_input_wav(self):
        logging.info(
            "\n\ntest19: This is a test to compress the data using the "
            + "'compress' method where the inputs are "
            + "sample_rate and the input_wav.\n\n"
        )
        sample_rate, input_wav = wavfile.read(filename=self.file)
        byte_string = encode.compress(sample_rate=sample_rate, input_wav=input_wav)
        self.assertEqual(type(byte_string), bytes)

    def test20_test_compress_sample_rate_input_wav_quick(self):
        logging.info(
            "\n\ntest20: This is a test to compress the data using the "
            + "'compress' method where the inputs are "
            + "sample_rate and input_wav while implementing the "
            + "'quick' option. \n\n"
        )
        sample_rate, input_wav = wavfile.read(filename=self.file)
        byte_string = encode.compress(
            sample_rate=sample_rate, input_wav=input_wav, quick=True
        )
        self.assertEqual(type(byte_string), bytes)

    def test21_test_main_method_of_compression_h(self):
        logging.info(
            "\n\ntest21: This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'h' which indicates a huffman encoding "
            + "format exclusively.\n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=h"])
        encode.main(args)

    def test22_test_main_method_of_compression_u(self):
        logging.info(
            "\n\ntest22: This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'u' which indicates implementing huffman encoding "
            + "and a unique amplitudes list.\n\n"
        )
        start_time = time.time_ns()
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=u"])
        encode.main(args)
        stop_time = time.time_ns()
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time,
            stop_time,
            executed_line="encode_using_amplitude_indices_less_than_256",
        )
        process_signal.print_size_of_file_compression(
            file_path=self.file, compressed_file_path=self.compressed_file_path
        )

    def test23_test_main_method_of_compression_n(self):
        logging.info(
            "\n\ntest23: This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'n' which indicates implementing neural spike "
            + "detection.\n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=n"])
        encode.main(args)

    def test24_test_compress_method_of_compression_u(self):
        logging.info(
            "\n\ntest24 This is a test to compress the data using the "
            + "'compress' method where the method of compression "
            + "will be interpreted to 'u' because the length of the "
            + "unique indices of the input amplitudes will be less "
            + "than 256 and the 'quick' option is set to 'False' "
            + "by default. \n\n"
        )

        start_time = time.time_ns()
        byte_string = encode.compress(file=self.file)
        stop_time = time.time_ns()

        self.assertEqual(type(byte_string), bytes)

        rate, data = wavfile.read(self.file)

        process_signal.print_compression_efficiency_metrics_wrapper(
            original_data=data,
            compressed_data=byte_string,
            start_time=start_time,
            stop_time=stop_time,
            method="encode.compress(file=self.file",
        )

    def test25_test_compress_method_of_compression_h(self):
        logging.info(
            "\n\ntest25: This is a test to compress the data using the "
            + "'compress' method where the method of compression "
            + "will be interpreted to 'h' because the length of the "
            + "unique indices of the input amplitudes will be more "
            + "than 256 and the 'quick' option is set to 'True'. \n\n"
        )

        start_time = time.time_ns()
        byte_string = encode.compress(file=self.debug_file, quick=True)
        stop_time = time.time_ns()

        self.assertEqual(type(byte_string), bytes)

        rate, data = wavfile.read(self.debug_file)

        process_signal.print_compression_efficiency_metrics_wrapper(
            original_data=data,
            compressed_data=byte_string,
            start_time=start_time,
            stop_time=stop_time,
            method="encode.compress(file=self.debug_file, quick=True)",
        )

    def test26_test_compress_method_of_compression_n(self):
        logging.info(
            "\n\ntest26: This is a test to compress the data using the "
            + "'compress' method where the method of compression "
            + "will be interpreted to 'n' because the length of the "
            + "unique indices of the input amplitudes will be more "
            + "than 256 and the 'quick' option is set to 'False' "
            + "by default. \n\n"
        )

        start_time = time.time_ns()
        byte_string = encode.compress(file=self.debug_file)
        stop_time = time.time_ns()

        self.assertEqual(type(byte_string), bytes)

        rate, data = wavfile.read(self.debug_file)

        process_signal.print_compression_efficiency_metrics_wrapper(
            original_data=data,
            compressed_data=byte_string,
            start_time=start_time,
            stop_time=stop_time,
            method="encode.compress(file=self.debug_file)",
        )

    def test27_test_main_method_of_compression_u_unique_greater_than_256(self):
        logging.info(
            "\n\ntest27: This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'u' which indicates implementing huffman encoding "
            + "and a unique amplitudes list. There are more than 256 "
            + "unique amplitudes in this list. \n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args(
            [self.debug_file, self.debug_compressed_file_path, "-m=u"]
        )
        encode.main(args)

    def test28_test_main_method_of_compression_n_unique_greater_than_256_quick_is_set(
        self,
    ):
        logging.info(
            "\n\ntest28: This is a test to compress the data using the "
            + " 'main' method where the method of compression "
            + "is 'n' which indicates implementing the "
            + "neural spike detection module. "
            + "The 'quick' parameter is set. This should not effect the "
            + "output size of the resultant data nor the method of "
            + "compression. \n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args(
            [self.debug_file, self.debug_compressed_file_path, "-m=n", "-q"]
        )
        encode.main(args)

        with open(self.debug_compressed_file_path, "rb+") as fp:
            debug_compressed_file_data = fp.read()
            fp.close()
        method_of_compression, huffman_encoded_data = (
            decode.extract_method_of_compression(debug_compressed_file_data)
        )
        self.assertEqual(method_of_compression, "n")

    def test29_test_main_method_of_compression_h_unique_greater_than_256_quick_is_set(
        self,
    ):
        logging.info(
            "\n\ntest29: This is a test to compress the data using the "
            + " 'main' method where the method of compression "
            + "is 'h' which indicates implementing the "
            + "huffman encoding algorithm. "
            + "The 'quick' parameter is set. This should not effect the "
            + "output size of the resultant data nor the method of "
            + "compression. \n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args(
            [self.debug_file, self.debug_compressed_file_path, "-m=h", "-q"]
        )
        encode.main(args)

        with open(self.debug_compressed_file_path, "rb+") as fp:
            debug_compressed_file_data = fp.read()
            fp.close()
        method_of_compression, huffman_encoded_data = (
            decode.extract_method_of_compression(debug_compressed_file_data)
        )
        self.assertEqual(method_of_compression, "h")

    def test30_test_main_method_of_compression_w_unique_greater_than_256_quick_is_set(
        self,
    ):
        logging.info(
            "\n\ntest30: This is a test to compress the data using the "
            + " 'main' method where the method of compression "
            + "is 'u' which indicates implementing the "
            + "huffman encoding algorithm with a unique amplitudes list. "
            + "The 'quick' parameter is set. This should not effect the "
            + "output size of the resultant data nor the method of "
            + "compression. The method of compression should be 'w' "
            + "because the there are more than 256 unique amplitudes in "
            + "the wav file titled 'debug_file' which contains "
            + "raw neural data amplitudes. \n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args(
            [self.debug_file, self.debug_compressed_file_path, "-m=u", "-q"]
        )
        encode.main(args)

        with open(self.debug_compressed_file_path, "rb+") as fp:
            debug_compressed_file_data = fp.read()
            fp.close()
        method_of_compression, huffman_encoded_data = (
            decode.extract_method_of_compression(debug_compressed_file_data)
        )
        self.assertEqual(method_of_compression, "w")

    def test31_test_main_method_of_compression_w_unique_greater_than_256_quick_is_set(
        self,
    ):
        logging.info(
            "\n\ntest31: This is a test to compress the data using the "
            + " 'main' method where the method of compression "
            + "is 'u' which indicates implementing the "
            + "huffman encoding algorithm with a unique amplitudes list. "
            + "The 'quick' parameter is set. This should not effect the "
            + "output size of the resultant data nor the method of "
            + "compression. The method of compression should be 'u' "
            + "because the there are less than 256 unique amplitudes in "
            + "the wav file titled 'file' which contains raw neural "
            + "data amplitudes. \n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=u", "-q"])
        encode.main(args)

        with open(self.compressed_file_path, "rb+") as fp:
            compressed_file_data = fp.read()
            fp.close()
        method_of_compression, huffman_encoded_data = (
            decode.extract_method_of_compression(compressed_file_data)
        )
        self.assertEqual(method_of_compression, "u")

    def test32_test_create_node_mapping_dictionary(self):
        logging.info(
            "This is a unittest of the function 'create_node_mapping_dictionary'. "
        )

        # Read Data for testing:
        sr, input_data = wavfile.read(self.file)

        """Establish input data for the create_node_mapping_dictionary
            function:
        """
        sorted_hex_freq_dict = encode.determine_hex_freq(
            input_data if type(input_data) == bytes else input_data.tobytes()
        )

        hex_freq_values = list(sorted_hex_freq_dict.values())
        hex_freq_keys = list(sorted_hex_freq_dict.keys())

        # Create a list of nodes
        nodes = []
        for item in range(len(hex_freq_keys)):
            heapq.heappush(
                nodes, encode.Node(hex_freq_values[item], hex_freq_keys[item])
            )

        # Build the node tree
        while len(nodes) > 1:
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)
            left.code = 0
            right.code = 1
            newNode = encode.Node(
                left.freq + right.freq, left.data + right.data, left=left, right=right
            )
        heapq.heappush(nodes, newNode)

        """Begin unit testing of function:
        """
        node_mapping_dict = encode.create_node_mapping_dictionary(
            nodes[0], val="", node_mapping_dict={}
        )
        self.assertTrue(node_mapping_dict)

    def test33_huffman_encoding(self):
        logging.info(
            "This is a test that the function "
            + "'huffman_encoding' properly functions."
        )
        sr, data = wavfile.read(self.file)
        node_mapping_dict, bit_string, end_zero_padding = encode.huffman_encoding(
            input_data=data
        )
        self.assertTrue(node_mapping_dict)
        self.assertGreater(len(node_mapping_dict), 1)
        self.assertTrue(bit_string)
        self.assertTrue(end_zero_padding)

    def test34_encode_using_amplitude_indices_less_than_256(self):
        logging.info(
            "This is a test that the function "
            + "'encode_using_amplitude_indices' "
            + "properly functions when the number of unique "
            + "amplitudes is less than 256."
        )

        sr, data_less_than_256_unique_amplitudes = wavfile.read(self.file)
        byte_string = encode.encode_using_amplitude_indices(
            data_less_than_256_unique_amplitudes
        )
        self.assertEqual(type(byte_string), bytes)

    def test35_encode_using_amplitude_indices_greater_than_256(self):
        logging.info(
            "This is a test that the function "
            + "'encode_using_amplitude_indices' "
            + "properly functions when the number of unique "
            + "amplitudes is greater than 256."
        )
        sr, data_greater_than_256_unique_amplitudes = wavfile.read(self.debug_file)
        byte_string = encode.encode_using_amplitude_indices(
            data_greater_than_256_unique_amplitudes
        )
        self.assertEqual(type(byte_string), bytes)

    def test36_test_unique_amplitudes_l_does_not_exist_and_method_of_compression_is_u_during_create_byte_string(
        self,
    ):
        logging.info(
            "\n\ntest36: This test for the case when the unique "
            + "amplitudes list ('unique_amplitudes_l') does not "
            + "exist and the method of compression is set to the "
            + "value of 'u' during the 'create_byte_string' "
            + "function.\n\n"
        )

        # Establishing data for the test:
        error_string_to_identify = (
            "Error unique_amplitudes_l does "
            + "not exist and method_of_compression_is_set to u. : "
            + "'NoneType' object has no attribute 'any'"
        )

        rate, data = wavfile.read(self.file)
        data_l = data.tolist()

        """There are less than or equal to 256 unique amplitudes.
            This indicates that an unsigned 8-bit ingeger will be used.
        """
        unique_amplitudes = np.unique(data_l).tolist()
        self.assertLess(len(unique_amplitudes), 256)
        indices = np.array(
            [unique_amplitudes.index(value) for value in data_l], dtype=np.uint8
        )

        node_mapping_dict, bit_string, end_zero_padding = encode.huffman_encoding(
            input_data=indices
        )

        """The unique_amplitudes_l (unique amplitudes list) is 
            erroneously not calculated nor sent in the function call
            below.
        """
        with self.assertRaises(Exception) as cm:
            byte_string = encode.create_byte_string(
                node_mapping_dict,
                bit_string,
                end_zero_padding,
                method_of_compression="u",
            )
        identified_exception = cm.exception
        self.assertEqual(identified_exception.args[0], error_string_to_identify)

    def test37_test_unique_amplitudes_l_does_not_exist_and_method_of_compression_is_w_during_create_byte_string(
        self,
    ):
        logging.info(
            "\n\ntest37: This test for the case when the unique "
            + "amplitudes list ('unique_amplitudes_l') does not "
            + "exist and the method of compression is set to the "
            + "value of 'w' during the 'create_byte_string' "
            + "function.\n\n"
        )

        # Establishing data for the test:
        error_string_to_identify = (
            "Error unique_amplitudes_l does "
            + "not exist and method_of_compression_is_set to w. : "
            + "'NoneType' object has no attribute 'any'"
        )

        rate, data = wavfile.read(self.debug_file)
        data_l = data.tolist()

        """There are more than 256 unique amplitudes.
            This indicates that an unsigned 16-bit integer will be used.
        """
        unique_amplitudes = np.unique(data_l).tolist()
        self.assertGreater(len(unique_amplitudes), 256)

        indices = np.array(
            [unique_amplitudes.index(value) for value in data_l], dtype=np.uint16
        )

        node_mapping_dict, bit_string, end_zero_padding = encode.huffman_encoding(
            input_data=indices
        )

        """The unique_amplitudes_l (unique amplitudes list) is 
            erroneously not calculated nor sent in the function call
            below.
        """
        with self.assertRaises(Exception) as cm:
            byte_string = encode.create_byte_string(
                node_mapping_dict,
                bit_string,
                end_zero_padding,
                method_of_compression="w",
            )
        identified_exception = cm.exception
        self.assertEqual(identified_exception.args[0], error_string_to_identify)

    def test38_test_unique_amplitudes_l_is_greater_than_65536_in_function_encode_using_amplitude_indices(
        self,
    ):
        logging.info(
            "\n\ntest38: This is a test for when the unique "
            + "amplitudes list is equal to or greater than 65536. "
            + "This case should raise a ValueError.\n\n"
        )

        # Creating Sample Data
        error_string_to_be_identified = (
            "The number of "
            + "unique_amplitudes is greater than 65536. The "
            + "indices will not be properly expressed by "
            + "unsigned 16-bit integers. Please select another "
            + "method of compression for this data."
        )
        # Values range from [0 to 65536] with list length of 65537.
        test_data = np.arange(0, 65537, step=1, dtype=np.int32)

        # Verifying a ValueError is raised:
        with self.assertRaises(ValueError) as cm:
            byte_string = encode.encode_using_amplitude_indices(data=test_data)

        # Verifying the ValueError message:
        identified_exception = cm.exception
        self.assertEqual(identified_exception.args[0], error_string_to_be_identified)

    def test39_test_file_not_defined_and_sample_rate_not_defined_in_compress_function(
        self,
    ):
        logging.info(
            "\n\ntest39: This test covers the case when the "
            + "file is not defined and the sample_rate is "
            + "not defined. "
        )
        error_string_to_be_identified = (
            "Error: compress requires either file "
            + "to be defined or sample_rate and "
            + "input_wav to be defined."
        )
        sample_rate, data = wavfile.read(self.file)

        with self.assertRaises(ValueError) as cm:
            byte_string = encode.compress(input_wav=data)
        identified_exception = cm.exception
        self.assertEqual(identified_exception.args[0], error_string_to_be_identified)

    def test40_test_file_not_defined_and_input_wav_not_defined_in_compress_function(
        self,
    ):
        logging.info(
            "\n\ntest40: This test covers the case when the "
            + "file is not defined and the input_wav (data) is "
            + "not defined. "
        )
        error_string_to_be_identified = (
            "Error: compress requires either file "
            + "to be defined or sample_rate and "
            + "input_wav to be defined."
        )
        sample_rate, data = wavfile.read(self.file)

        with self.assertRaises(ValueError) as cm:
            byte_string = encode.compress(sample_rate=sample_rate)
        identified_exception = cm.exception
        self.assertEqual(identified_exception.args[0], error_string_to_be_identified)

    def test41_test_compress_method_is_u_and_unique_amplitude_indices_is_less_than_257(self):
        logging.info("\n\ntest41: This tests the case that the method "
                     + "of compression is 'u' and the number of unique "
                     + "amplitude indices is less than 257 when the "
                     + "'compress' function is implemented. \n\n")
        byte_string = encode.compress(file=self.file, method='u')
        self.assertEqual(type(byte_string), bytes)

    def test42_test_compress_method_is_u_and_unique_amplitude_indices_is_greater_than_256_and_less_than_65537(self):
        logging.info("\n\ntest42: This tests the case that the method "
                     + "of compression is 'u' and the unique amplitude "
                     + "indices are between 257 and 65536 (inclusively "
                     + "at both extremes) during the 'compress' "
                     + "function execution.\n\n")
        byte_string = encode.compress(file = self.debug_file, method='u')
        self.assertEqual(type(byte_string), bytes)
        
    def test43_test_compress_method_is_n(self):
        logging.info("\n\ntest43: This tests the case during the "
                    + "'compress' function that the method of "
                    + "compression is 'n'.")
        byte_string = encode.compress(file = self.file, method='n')
        self.assertEqual(type(byte_string), bytes)

    def test44_test_compress_method_value_error(self):
        logging.info("\n\ntest44: This tests the case that the method "
                    + "of compression is not of value 'h', 'u', or "
                    + "'n'.")
        expected_error_string = ( 
                "Error: the method of compression (method) " 
                + "must be of value 'u', 'h', or 'n'. " 
                + "Please select a new value for the method of " 
                + "compression.") 
        with self.assertRaises(ValueError) as cm:
            byte_string = encode.compress(file = self.file, method = 'x')
        identified_exception = cm.exception
        self.assertEqual(identified_exception.args[0], expected_error_string)

    def test45_test_method_of_compression_is_quick_during_compess(self):
        logging.info("\n\ntest45: This tests the case during the "
                     + "'compress' method that the 'quick' option is "
                     + "sent and the number of unique amplitudes is " 
                     + "greater than 65536.\n\n")
        byte_string = encode.compress(sample_rate = self.test_sample_rate, input_wav = self.test_data, quick=True)
        self.assertEqual(type(byte_string), bytes)
        
    def test46_test_method_of_compression_is_not_quick_during_compress(self):
        logging.info("\n\ntest46: This tests the case during the "
                     + "'compress' method that the 'quick' option is "
                     + "not set and the number of unique amplitudes is "
                     + "greater than 65536. This will enable "
                     + "the method of compression to implement the "
                     + "neural spike detection module.\n\n")
        
        byte_string = encode.compress(sample_rate = self.test_sample_rate, input_wav = self.test_data)

    def test47_test_main_length_of_unique_amplitudes_is_less_than_65537(self):
        logging.info("\n\ntest47: This tests the 'main' function in "
                    + "the case that the number of unique amplitudes "
                    + "is less than 65537 and the quick argument is "
                    + "not defined.\n\n")
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path])
        encode.main(args=args)
        byte_string = process_signal.read_file_bytes(file_path=self.compressed_file_path)
        self.assertEqual(type(byte_string), bytes)
        encoded_method_of_compression = byte_string[:-1].decode(encoding='utf-8')
        self.assertEqual(encoded_method_of_compression, 'n')

    def test48_test_main_length_of_unique_amplitudes_is_greater_than_65536_quick(self):
        logging.info("\n\ntest48: This tests the 'main' function in "
                     + "the case that the number of unique amplitudes "
                     + "is greater than 65536 and the quick argument "
                     + "is set to 'True'. \n\n")
        # Create Sample Test File
        wavfile.write(filename=self.test_file_path, rate = self.test_sample_rate, data = self.test_data)
        
        # Encode the Sample Test File
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.test_file_path, self.test_compressed_file_path, '-q'])
        encode.main(args=args)
        
        # Assert the Encoding Properly Functioned
        byte_string = process_signal.read_file_bytes(file_path=self.compressed_file_path)
        self.assertEqual(type(byte_string), bytes)

        encoded_method_of_compression = byte_string[:-1].decode(encoding='utf-8')
        self.assertEqual(encoded_method_of_compression, 'h')

    def test49_test_main_length_of_unique_amplitudes_is_greater_than_65536_neural_spike_detection_implemented(self):
        logging.info("\n\ntest49: This tests the 'main' function in "
                     + "the case that the number of unique amplitudes "
                     + "is greater than 65536 and the quick argument "
                     + "is set to 'False'. \n\n")
        # Create Sample Test File
        wavfile.write(filename=self.test_file_path, rate = self.test_sample_rate, data = self.test_data)
        
        # Encode the Sample Test File
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.test_file_path, self.test_compressed_file_path])
        encode.main(args=args)
        
        # Assert the Encoding Properly Functioned
        byte_string = process_signal.read_file_bytes(file_path=self.compressed_file_path)
        self.assertEqual(type(byte_string), bytes)

        encoded_method_of_compression = byte_string[:-1].decode(encoding='utf-8')
        self.assertEqual(encoded_method_of_compression, 'n')


if __name__ == "__main__":
    unittest.main()
