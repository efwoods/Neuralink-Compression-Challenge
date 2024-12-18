"""This module tests the decoding module named decode."""

import unittest
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
import logging
import numpy as np
import time
from signal_processing_utilities import process_signal
from scipy.io import wavfile


# Set logging to all logging levels
logging.basicConfig(level=logging.DEBUG)

# Custom import of local file "decode"
spec = spec_from_loader("decode", SourceFileLoader("decode", "./decode"))
decode = module_from_spec(spec)
spec.loader.exec_module(decode)

# Custom import of local file "encode"
spec = spec_from_loader("encode", SourceFileLoader("encode", "./encode"))
encode = module_from_spec(spec)
spec.loader.exec_module(encode)


class TestDecode(unittest.TestCase):
    """This class contains test cases for the decode module.

    Args:
        unittest (module): This module enables custom tests.
    """

    def setUp(self):
        self.file = "data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav"
        self.compressed_file_path = (
            "data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav.brainwire"
        )
        self.decompressed_file_path = (
            "data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav.copy"
        )
        self.debug_file = "data/0052503c-2849-4f41-ab51-db382103690c.wav"
        self.debug_compressed_file_path = (
            "data/0052503c-2849-4f41-ab51-db382103690c.wav.brainwire"
        )
        self.debug_decompressed_file_path = (
            "data/0052503c-2849-4f41-ab51-db382103690c.wav.copy"
        )
        # The sample rate is implicitly a known value when exclusively
        # performing huffman compression.
        self.sample_rate = 19531

    def tearDown(self) -> None:
        pass

    def test01_huffman_decoding(self):
        logging.info(
            "\n\ntest01: This is a test of decoding a huffman "
            + "encoded file exclusively. \n\n"
        )

        # Creating Test Data
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-q"])
        sample_rate, input_wav = wavfile.read(args.file_path)
        byte_string = encode.create_huffman_encoded_file(input_wav)
        process_signal.write_file_bytes(
            file_path=args.compressed_file_path, data_bytes=byte_string
        )

        # Reading from the file and decoding
        huffman_encoded_data = decode.read_encoded_file(
            compressed_file_path=self.compressed_file_path,
        )
        method_of_compression, huffman_encoded_data = (
            decode.extract_method_of_compression(huffman_encoded_data)
        )
        decoded_wav_bytes, _ = decode.huffman_decoding(huffman_encoded_data)

    def test02_huffman_decoding_to_encoded_format(self):
        logging.info(
            "\n\ntest02: This is a test to decode the huffman encoded byte string,"
            + " convert the byte string into the encoded format, and"
            + " reconstruct the amplitude array.\n\n"
        )
        # Creating Test Data
        parser = encode.initialize_argument_parser()
        args = parser.parse_args(
            [
                self.file,
                self.compressed_file_path,
            ]
        )
        sample_rate, input_wav = wavfile.read(filename=args.file_path)
        byte_string = encode.implement_spike_detection_module_and_huffman_encode_file(
            sample_rate, input_wav
        )
        process_signal.write_file_bytes(
            file_path=args.compressed_file_path, data_bytes=byte_string
        )

        huffman_encoded_data = decode.read_encoded_file(
            compressed_file_path=self.compressed_file_path,
        )

        method_of_compression, huffman_encoded_data = (
            decode.extract_method_of_compression(huffman_encoded_data)
        )
        decoded_wav_bytes, _ = decode.huffman_decoding(huffman_encoded_data)

        #
        encoded_data = process_signal.convert_byte_string_to_encoded_data(
            encoded_data_byte_string=decoded_wav_bytes
        )
        sample_rate, amplitude_array = process_signal.decode_data(encoded_data)

        #
        amplitude_array = np.frombuffer(decoded_wav_bytes, dtype=np.int16)
        wavfile.write(
            filename=self.decompressed_file_path,
            rate=sample_rate,
            data=amplitude_array,
        )

    def test03_decoding_encoded_byte_string(self):
        logging.info(
            "\n\ntest03: This test encodes a file using huffman "
            + "encoding and decodes using huffman encoding.\n\n"
        )
        # Test 06 from test_encode.py of Huffman Encoding
        encoding_start_time = time.time_ns()
        sample_rate, input_wav = wavfile.read(self.file)
        node_mapping_dict, bit_string, end_zero_padding = encode.huffman_encoding(
            input_data=input_wav
        )

        byte_string = encode.create_byte_string(
            node_mapping_dict, bit_string, end_zero_padding, method_of_compression="h"
        )
        process_signal.write_file_bytes(
            file_path=self.compressed_file_path, data_bytes=byte_string
        )

        encoding_stop_time = time.time_ns()

        print("Encoding Time: ")
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=encoding_start_time, stop_time=encoding_stop_time
        )

        # Decoding Below
        # decoding_start_time = time.time_ns()
        encoded_data_byte_string = decode.read_encoded_file(
            compressed_file_path=self.compressed_file_path,
        )

        method_of_compression, encoded_data_byte_string = (
            decode.extract_method_of_compression(encoded_data_byte_string)
        )
        decoded_wav_bytes, _ = decode.huffman_decoding(encoded_data_byte_string)

        amplitude_array = np.frombuffer(decoded_wav_bytes, dtype=np.int16)

        wavfile.write(
            filename=self.decompressed_file_path,
            rate=self.sample_rate,
            data=amplitude_array,
        )

        decoding_stop_time = time.time_ns()

        # print("Decoding Time: ")
        # process_signal.print_time_each_function_takes_to_complete_processing(
        #     start_time=decoding_start_time, stop_time=decoding_stop_time
        # )
        print("Total Time: ")
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=encoding_start_time, stop_time=decoding_stop_time
        )
        process_signal.print_size_of_file_compression(
            file_path=self.file, compressed_file_path=self.compressed_file_path
        )

    def test04_huffman_decode_operates(self):
        logging.info("\n\ntest04: Testing Huffman Decoding exclusively.\n\n")
        huffman_encoded_string = decode.read_encoded_file(
            compressed_file_path=self.compressed_file_path
        )

        method_of_compression, huffman_encoded_string = (
            decode.extract_method_of_compression(huffman_encoded_string)
        )
        decoded_wav_bytes, _ = decode.huffman_decoding(
            huffman_encoded_data=huffman_encoded_string
        )

        wavfile.write(
            filename=self.decompressed_file_path,
            rate=self.sample_rate,
            data=np.frombuffer(decoded_wav_bytes, dtype=np.int16),
        )

    def test05_encode_data_implement_huffman_encoding_and_decode(self):
        logging.info(
            "\n\ntest05: This is an end-to-end test of the huffman encode & decode "
            + "algorithm with signal processing included.\n\n"
        )

        logging.info(
            "This is a test to encode the huffman encoded byte string,"
            + " convert the byte string into the encoded format, and"
            + " reconstruct the amplitude array."
        )

        # Spike Detection & Huffman Encoding
        encoding_start_time = time.time_ns()

        sample_rate, input_wav = wavfile.read(self.file)
        filtered_data_bandpass = process_signal.preprocess_signal(
            raw_neural_signal=input_wav,
            sample_rate=sample_rate,
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
            file_path=self.compressed_file_path, data_bytes=byte_string
        )

        encoding_stop_time = time.time_ns()

        print("Encoding Time: ")
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=encoding_start_time, stop_time=encoding_stop_time
        )

        # Decoding
        decoding_start_time = time.time_ns()
        huffman_encoded_data = decode.read_encoded_file(
            compressed_file_path=self.compressed_file_path,
        )

        method_of_compression, huffman_encoded_data = (
            decode.extract_method_of_compression(huffman_encoded_data)
        )
        decoded_wav_bytes, _ = decode.huffman_decoding(huffman_encoded_data)
        encoded_data = process_signal.convert_byte_string_to_encoded_data(
            encoded_data_byte_string=decoded_wav_bytes
        )
        sample_rate, amplitude_array = process_signal.decode_data(encoded_data)

        wavfile.write(
            filename=self.decompressed_file_path,
            rate=sample_rate,
            data=amplitude_array,
        )

        decoding_stop_time = time.time_ns()

        print("Decoding Time: ")
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=decoding_start_time, stop_time=decoding_stop_time
        )
        print("Total Time: ")
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=encoding_start_time, stop_time=decoding_stop_time
        )
        process_signal.print_size_of_file_compression(
            file_path=self.file, compressed_file_path=self.compressed_file_path
        )

    def test06_encode_data_implement_huffman_encoding_and_decode(self):
        logging.info(
            "\n\ntest06: This is a test to encode the huffman encoded byte string,"
            + " convert the byte string into the encoded format, and"
            + " reconstruct the amplitude array.\n\n"
        )

        logging.info("Debugging Key:Value Pair not found in bit string.")
        logging.info("Debugging: decode.find_key_by_value_in_node_mapping_dictionary(")

        logging.info(
            "Debugging the following error: \n"
            + "ValueError: setting an array element with a"
            + "sequence. The requested array has an "
            + "inhomogeneous shape after 1 dimensions. The "
            + "detected shape was (3850,) + inhomogeneous "
            + "part. Traceback (most recent call last)"
        )

        # Spike Detection & Huffman Encoding
        encoding_start_time = time.time_ns()
        sample_rate, input_wav = wavfile.read(
            self.debug_file,
        )
        filtered_data_bandpass = process_signal.preprocess_signal(
            raw_neural_signal=input_wav,
            sample_rate=sample_rate,
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
        encoding_stop_time = time.time_ns()

        byte_string = encode.create_byte_string(
            node_mapping_dict, bit_string, end_zero_padding, method_of_compression="n"
        )

        process_signal.write_file_bytes(
            file_path=self.debug_compressed_file_path, data_bytes=byte_string
        )

        # Decoding
        decoding_start_time = time.time_ns()
        huffman_encoded_data = decode.read_encoded_file(
            compressed_file_path=self.compressed_file_path,
        )

        method_of_compression, huffman_encoded_data = (
            decode.extract_method_of_compression(huffman_encoded_data)
        )
        decoded_wav_bytes, _ = decode.huffman_decoding(huffman_encoded_data)
        encoded_data = process_signal.convert_byte_string_to_encoded_data(
            encoded_data_byte_string=decoded_wav_bytes
        )

        sample_rate, amplitude_array = process_signal.decode_data(encoded_data)
        wavfile.write(
            filename=self.decompressed_file_path,
            rate=sample_rate,
            data=amplitude_array,
        )
        decoding_stop_time = time.time_ns()

        print(f"Encoding Time:")
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=encoding_start_time, stop_time=encoding_stop_time
        )
        print(f"Decoding Time:")
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=decoding_start_time, stop_time=decoding_stop_time
        )
        print(f"Total Time:")
        process_signal.print_time_each_function_takes_to_complete_processing(
            start_time=encoding_start_time, stop_time=decoding_stop_time
        )

        process_signal.print_size_of_file_compression(
            file_path=self.file, compressed_file_path=self.compressed_file_path
        )

    def test07_test_of_arg_parser(self):
        logging.info(
            "\n\ntest07: This is a test hat the parser is "
            + "appropriately initialized and the arguments "
            + "are successfully parsed.\n\n"
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

    def test08_test_decompressing_compress_file_name(self):
        logging.info(
            "\n\ntest08: This is a test to compress the data using the "
            + "'compress' method and the file name. "
            + "The data is then decompressed.\n\n"
        )
        logging.info("Method of compression == 'u'")
        byte_string = encode.compress(file=self.file)

        self.assertEqual(type(byte_string), bytes)

        process_signal.write_file_bytes(
            file_path=self.compressed_file_path, data_bytes=byte_string
        )

        parser = decode.initialize_argument_parser()
        args = parser.parse_args(
            [self.compressed_file_path, self.decompressed_file_path]
        )

        decode.main(args)
        sample_rate, data = wavfile.read(self.decompressed_file_path)

        self.assertEqual(type(sample_rate), int)
        self.assertEqual(sample_rate, 19531)
        self.assertEqual(type(data), np.ndarray)

        original_sample_rate, original_data = wavfile.read(self.file)

        # Verifying the amplitudes are equivalent.
        all_values_equal = True
        for index, value in enumerate(original_data):
            if value != data[index]:
                all_values_equal = False
                print(f"decompressed value not equal to original value")
                print(f"Index: {[index]}")
                print(f"decompressed value: {data[index]}")
                print(f"original value: {value}")
        self.assertTrue(all_values_equal)
        if all_values_equal:
            print("All values between original amplitudes and ", end="")
            print("decompressed amplitudes are equivalent.")

    def test09_test_decompress_compress_file_name_quick(self):
        logging.info(
            "\n\ntest09: This is a test to compress the data using the "
            + "'compress' method and the file name where "
            + "the quick argument is passed into the function. "
            + "The data is then decompressed.\n\n"
        )
        byte_string = encode.compress(file=self.file, quick=True)
        self.assertEqual(type(byte_string), bytes)

        process_signal.write_file_bytes(
            file_path=self.compressed_file_path, data_bytes=byte_string
        )

        parser = decode.initialize_argument_parser()
        args = parser.parse_args(
            [self.compressed_file_path, self.decompressed_file_path]
        )

        decode.main(args)
        sample_rate, data = wavfile.read(self.decompressed_file_path)

        self.assertEqual(type(sample_rate), int)
        self.assertEqual(sample_rate, 19531)
        self.assertEqual(type(data), np.ndarray)

        original_sample_rate, original_data = wavfile.read(self.file)

        # Verifying the amplitudes are equivalent.
        all_values_equal = True
        for index, value in enumerate(original_data):
            if value != data[index]:
                all_values_equal = False
                print(f"decompressed value not equal to original value")
                print(f"Index: {[index]}")
                print(f"decompressed value: {data[index]}")
                print(f"original value: {value}")
        self.assertTrue(all_values_equal)
        if all_values_equal:
            print("All values between original amplitudes and ", end="")
            print("decompressed amplitudes are equivalent.")

    def test10_test_compress_sample_rate_input_wav(self):
        logging.info(
            "\n\ntest10: This is a test to compress the data using the "
            + "'compress' method where the inputs are "
            + "sample_rate and the input_wav. "
            + "The data is then decompressed. "
            + "Method of compression == 'u'.\n\n"
        )

        sample_rate, input_wav = wavfile.read(filename=self.file)
        byte_string = encode.compress(sample_rate=sample_rate, input_wav=input_wav)
        self.assertEqual(type(byte_string), bytes)

        process_signal.write_file_bytes(
            file_path=self.compressed_file_path, data_bytes=byte_string
        )

        parser = decode.initialize_argument_parser()
        args = parser.parse_args(
            [self.compressed_file_path, self.decompressed_file_path]
        )

        decode.main(args)
        sample_rate, data = wavfile.read(self.decompressed_file_path)

        self.assertEqual(type(sample_rate), int)
        self.assertEqual(sample_rate, 19531)
        self.assertEqual(type(data), np.ndarray)

        original_sample_rate, original_data = wavfile.read(self.file)

        # Verifying the amplitudes are equivalent.
        all_values_equal = True
        for index, value in enumerate(original_data):
            if value != data[index]:
                all_values_equal = False
                print(f"decompressed value not equal to original value")
                print(f"Index: {[index]}")
                print(f"decompressed value: {data[index]}")
                print(f"original value: {value}")
        self.assertTrue(all_values_equal)
        if all_values_equal:
            print("All values between original amplitudes and ", end="")
            print("decompressed amplitudes are equivalent.")

    def test11_test_decompress_compress_sample_rate_input_wav_quick(self):
        logging.info(
            "\n\ntest11: This is a test to compress the data using the "
            + "'compress' method where the inputs are "
            + "sample_rate and the input_wav. "
            + "The data is then decompressed. "
            + "Method of compression is quick.\n\n"
        )
        sample_rate, input_wav = wavfile.read(filename=self.file)
        byte_string = encode.compress(
            sample_rate=sample_rate, input_wav=input_wav, quick=True
        )
        self.assertEqual(type(byte_string), bytes)

        process_signal.write_file_bytes(
            file_path=self.compressed_file_path, data_bytes=byte_string
        )

        parser = decode.initialize_argument_parser()
        args = parser.parse_args(
            [self.compressed_file_path, self.decompressed_file_path]
        )

        decode.main(args)
        sample_rate, data = wavfile.read(self.decompressed_file_path)

        self.assertEqual(type(sample_rate), int)
        self.assertEqual(sample_rate, 19531)
        self.assertEqual(type(data), np.ndarray)

        original_sample_rate, original_data = wavfile.read(self.file)

        # Verifying the amplitudes are equivalent.
        all_values_equal = True
        for index, value in enumerate(original_data):
            if value != data[index]:
                all_values_equal = False
                print(f"decompressed value not equal to original value")
                print(f"Index: {[index]}")
                print(f"decompressed value: {data[index]}")
                print(f"original value: {value}")
        self.assertTrue(all_values_equal)
        if all_values_equal:
            print("All values between original amplitudes and ", end="")
            print("decompressed amplitudes are equivalent.")

    def test12_test_decompress_main_method_of_compression_h(self):
        logging.info(
            "\n\ntest12: This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'h' which indicates a huffman encoding "
            + "format exclusively. "
            + "The data is then decompressed. "
            + "The method of compression is 'h'. \n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=h"])
        encode.main(args)

        parser = decode.initialize_argument_parser()
        args = parser.parse_args(
            [self.compressed_file_path, self.decompressed_file_path]
        )

        decode.main(args)
        sample_rate, data = wavfile.read(self.decompressed_file_path)

        self.assertEqual(type(sample_rate), int)
        self.assertEqual(sample_rate, 19531)
        self.assertEqual(type(data), np.ndarray)

        original_sample_rate, original_data = wavfile.read(self.file)

        # Verifying the amplitudes are equivalent.
        all_values_equal = True
        for index, value in enumerate(original_data):
            if value != data[index]:
                all_values_equal = False
                print(f"decompressed value not equal to original value")
                print(f"Index: {[index]}")
                print(f"decompressed value: {data[index]}")
                print(f"original value: {value}")
        self.assertTrue(all_values_equal)
        if all_values_equal:
            print("All values between original amplitudes and ", end="")
            print("decompressed amplitudes are equivalent.")

    def test13_test_decompress_main_method_of_compression_u(self):
        logging.info(
            "\n\ntest13: This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'h' which indicates a huffman encoding "
            + "format exclusively. "
            + "The data is then decompressed. "
            + "The method of compression is 'u'. \n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=u"])
        encode.main(args)

        parser = decode.initialize_argument_parser()
        args = parser.parse_args(
            [self.compressed_file_path, self.decompressed_file_path]
        )

        decode.main(args)
        sample_rate, data = wavfile.read(self.decompressed_file_path)

        self.assertEqual(type(sample_rate), int)
        self.assertEqual(sample_rate, 19531)
        self.assertEqual(type(data), np.ndarray)

        original_sample_rate, original_data = wavfile.read(self.file)

        # Verifying the amplitudes are equivalent.
        all_values_equal = True
        for index, value in enumerate(original_data):
            if value != data[index]:
                all_values_equal = False
                print(f"decompressed value not equal to original value")
                print(f"Index: {[index]}")
                print(f"decompressed value: {data[index]}")
                print(f"original value: {value}")
        self.assertTrue(all_values_equal)
        if all_values_equal:
            print("All values between original amplitudes and ", end="")
            print("decompressed amplitudes are equivalent.")

    def test14_test_decompress_main_method_of_compression_n_for_functionality(self):
        logging.info(
            "\n\ntest14: This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'n' which indicates implementing neural spike "
            + "detection. "
            + "The data is then decompressed. "
            + "The method of compression is 'n'. \n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=n"])
        encode.main(args)

        parser = decode.initialize_argument_parser()
        args = parser.parse_args(
            [self.compressed_file_path, self.decompressed_file_path]
        )

        decode.main(args)
        sample_rate, data = wavfile.read(self.decompressed_file_path)

        self.assertEqual(type(sample_rate), int)
        self.assertEqual(sample_rate, 19531)
        self.assertEqual(type(data), np.ndarray)

    def test15_test_decompress_main_method_of_compression_n_for_equivalency(self):
        logging.info(
            "\n\ntest15: This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'n' which indicates implementing neural spike "
            + "detection. "
            + "The data is then decompressed. "
            + "The method of compression is 'n'. \n\n"
        )
        logging.warning(
            "\nWhen the method of compression == 'n', the "
            + "original data will not match the filtered data "
            + "identically.\n"
        )
        # Encoding The Data
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=n"])
        encode.main(args)

        # Decoding The Data
        parser = decode.initialize_argument_parser()
        args = parser.parse_args(
            [self.compressed_file_path, self.decompressed_file_path]
        )
        decode.main(args)

        sample_rate, data = wavfile.read(self.decompressed_file_path)

        # Verifying the decoded data
        self.assertEqual(type(sample_rate), int)
        self.assertEqual(sample_rate, 19531)
        self.assertEqual(type(data), np.ndarray)

        original_sample_rate, original_data = wavfile.read(self.file)

        # Verifying the amplitudes are not equivalent.
        self.assertFalse(original_data.all() == data.all())

        # Explicitly defining where the amplitudes are not equivalent.
        all_values_equal = True
        for index, value in enumerate(original_data):
            if value != data[index]:
                all_values_equal = False
                # print(f"decompressed value not equal to original value")
                # print(f"Index: {[index]}")
                # print(f"decompressed value: {data[index]}")
                # print(f"original value: {value}")

        # When the method of compression == 'n', the
        # original data will not match the filtered data identically.
        self.assertFalse(all_values_equal)

        if all_values_equal:
            print("All values between original amplitudes and ", end="")
            print("decompressed amplitudes are equivalent.")

    def test16_test_decompress_method_of_compression_u(self):
        logging.info(
            "\n\ntest16: This is a test to compress the data using the "
            + "'compress' method where the method of compression "
            + "will be interpreted to 'u' because the length of the "
            + "unique indices of the input amplitudes will be less "
            + "than 256 and the 'quick' option is set to 'False' "
            + "by default. \n\n"
        )

        byte_string = encode.compress(file=self.file)
        rate, data = decode.decompress(byte_string=byte_string)

        self.assertEqual(rate, 19531)
        self.assertEqual(type(data), np.ndarray)

    def test17_test_decompress_method_of_compression_h(self):
        logging.info(
            "\n\ntest17: This is a test to compress the data using the "
            + "'compress' method where the method of compression "
            + "will be interpreted to 'h' because the length of the "
            + "unique indices of the input amplitudes will be more "
            + "than 256 and the 'quick' option is set to 'True' "
            + "by default. \n\n"
        )

        byte_string = encode.compress(file=self.debug_file, quick=True)
        rate, data = decode.decompress(byte_string=byte_string)

        self.assertEqual(rate, 19531)
        self.assertEqual(type(data), np.ndarray)

    def test18_test_decompress_method_of_compression_n(self):
        logging.info(
            "\n\ntest18: This is a test to compress the data using the "
            + "'compress' method where the method of compression "
            + "will be interpreted to 'n' because the length of the "
            + "unique indices of the input amplitudes will be more "
            + "than 256 and the 'quick' option is set to 'False' "
            + "by default. \n\n"
        )

        byte_string = encode.compress(file=self.debug_file)
        rate, data = decode.decompress(byte_string=byte_string)

        self.assertEqual(rate, 19531)
        self.assertEqual(type(data), np.ndarray)

    def test19_test_process_huffman_encoded_file_return_value_from_decompress_method(
        self,
    ):
        logging.info(
            "\n\ntest19: This method tests that the function "
            + "'process_huffman_encoded_file' will "
            + "properly return a value within the "
            + "decompress method. This is black box testing "
            + "of the decompress function within decode.py.\n\n"
        )
        # Encode using huffman encoding
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=h"])
        encode.main(args=args)
        byte_string = process_signal.read_file_bytes(self.compressed_file_path)
        # Assert that the method of compression is huffman encoding:
        self.assertEqual(byte_string[-1:].decode(encoding="utf-8"), "h")

        # Decompress the data
        rate, data = decode.decompress(byte_string)
        self.assertEqual(rate, 19531)
        self.assertEqual(data.dtype, "int16")
        self.assertEqual(type(data), np.ndarray)

    def test20_test_process_neural_spike_detection_compression_of_decompress(self):
        logging.info(
            "\n\ntest20: This is a test of the "
            + "'process_spike_detection_huffman_encoded_data' "
            + "function in the decompress function of "
            + "decode.py.\n\n"
        )
        # Encode using the neural spike detection module
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=n"])
        encode.main(args=args)
        byte_string = process_signal.read_file_bytes(self.compressed_file_path)
        # Assert the method of compression is type 'n':
        self.assertEqual(byte_string[-1:].decode(encoding="utf-8"), "n")

        # Decode the data
        rate, data = decode.decompress(byte_string)
        self.assertEqual(rate, 19531)
        self.assertEqual(data.dtype, "int16")
        self.assertEqual(type(data), np.ndarray)

    def test21_test_error_of_method_of_compression_within_decompress(self):
        logging.info(
            "\n\ntest21: This is a test to ensure that the "
            + "decompress method will raise an error if "
            + "the method of compression is not 'h', 'u', "
            + "'w', or 'n'.\n\n"
        )
        parser = encode.initialize_argument_parser()
        args = parser.parse_args([self.file, self.compressed_file_path, "-m=n"])
        encode.main(args=args)
        byte_string = process_signal.read_file_bytes(self.compressed_file_path)

        # Modifying byte_string to contain an error for the method of compression
        byte_string = byte_string[:-1]
        byte_string += b"x"
        self.assertEqual(byte_string[-1:].decode(encoding="utf-8"), "x")

        # Verify this condition raises a Value Error
        with self.assertRaises(ValueError) as cm:
            rate, data = decode.decompress(byte_string)
        identified_exception = cm.exception

        # Verify the ValueError is because of the method of compression.
        self.assertEqual(
            identified_exception.args[0],
            "Method of compression is not 'h', 'u', 'w' or 'n'.",
        )

    def test22_test_main_will_raise_value_error_with_improper_method_of_compression_in_byte_string(
        self,
    ):
        logging.info(
            "\n\ntest22: This test will create an incorrect method of "
            + "compression that is not accepted before running the "
            + "main function to detect if a ValueError detects this "
            + "error.\n\n"
        )
        # Initialize encoded data
        encode_parser = encode.initialize_argument_parser()
        encode_args = encode_parser.parse_args(
            [self.file, self.compressed_file_path, "-m=n"]
        )
        encode.main(args=encode_args)
        byte_string = process_signal.read_file_bytes(self.compressed_file_path)

        # Modifying byte_string to contain an error for the
        #   method of compression:
        byte_string = byte_string[:-1]
        byte_string += b"x"
        self.assertEqual(byte_string[-1:].decode(encoding="utf-8"), "x")

        # Overwrite the compressed_file_path with the erroneous data
        with open(self.compressed_file_path, "wb") as cfp:
            cfp.write(byte_string)
            cfp.close()

        # Verify that the erroneous data has been overwritten:
        modified_byte_string = process_signal.read_file_bytes(self.compressed_file_path)
        self.assertEqual(modified_byte_string[-1:].decode(encoding="utf-8"), "x")

        # Verify this condition will raise an error in main:
        decode_parser = decode.initialize_argument_parser()
        decode_args = decode_parser.parse_args(
            [self.compressed_file_path, self.decompressed_file_path]
        )
        with self.assertRaises(ValueError) as cm:
            decode.main(args=decode_args)
        identified_exception = cm.exception
        self.assertEqual(
            identified_exception.args[0],
            "Method of compression is not 'h', 'u', or 'n'.",
        )


if __name__ == "__main__":
    unittest.main()
