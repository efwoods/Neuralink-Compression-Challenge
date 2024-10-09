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
        logging.info("This is a test of decoding a huffman encoded file exclusively")

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
            "This is a test to decode the huffman encoded byte string,"
            + " convert the byte string into the encoded format, and"
            + " reconstruct the amplitude array."
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
            "This test encodes a file using huffman encoding and decodes using huffman encoding."
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
        logging.info("Testing Huffman Decoding exclusively.")
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
            "This is an end-to-end test of the huffman encode & decode "
            + "algorithm with signal processing included."
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
        logging.info("Debugging Key:Value Pair not found in bit string.")
        logging.info("Debugging: decode.find_key_by_value_in_node_mapping_dictionary(")

        logging.info(
            "This is a test to encode the huffman encoded byte string,"
            + " convert the byte string into the encoded format, and"
            + " reconstruct the amplitude array."
        )

        logging.info(
            "ValueError: setting an array element with a"
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
            "This is a test to compress the data using the "
            + "'compress' method and the file name. "
            + "The data is then decompressed."
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
            "This is a test to compress the data using the "
            + "'compress' method and the file name where "
            + "the quick argument is passed into the function. "
            + "The data is then decompressed."
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
            "This is a test to compress the data using the "
            + "'compress' method where the inputs are "
            + "sample_rate and the input_wav. "
            + "The data is then decompressed. "
            + "Method of compression == 'u'."
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
            "This is a test to compress the data using the "
            + "'compress' method where the inputs are "
            + "sample_rate and the input_wav. "
            + "The data is then decompressed. "
            + "Method of compression is quick."
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

    def test12_test_decompress_main_method_of_compression_q(self):
        logging.info(
            "This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'h' which indicates a huffman encoding "
            + "format exclusively. "
            + "The data is then decompressed. "
            + "The method of compression is 'h'. "
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
            "This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'h' which indicates a huffman encoding "
            + "format exclusively. "
            + "The data is then decompressed. "
            + "The method of compression is 'u'. "
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
            "This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'n' which indicates implementing neural spike "
            + "detection. "
            + "The data is then decompressed. "
            + "The method of compression is 'n'. "
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

    @unittest.skip(
        "When the method of compression == 'n', the "
        + "original data will not match the filtered data."
    )
    def test14_test_decompress_main_method_of_compression_n_for_equivalency(self):
        logging.info(
            "This is a test to compress the data using the "
            + "'main' method where the method of compression "
            + "is 'n' which indicates implementing neural spike "
            + "detection. "
            + "The data is then decompressed. "
            + "The method of compression is 'n'. "
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

        original_sample_rate, original_data = wavfile.read(self.file)

        # Verifying the amplitudes are equivalent.
        all_values_equal = True
        for index, value in enumerate(original_data):
            if value != data[index]:
                all_values_equal = False
                # print(f"decompressed value not equal to original value")
                # print(f"Index: {[index]}")
                # print(f"decompressed value: {data[index]}")
                # print(f"original value: {value}")
        self.assertTrue(all_values_equal)
        if all_values_equal:
            print("All values between original amplitudes and ", end="")
            print("decompressed amplitudes are equivalent.")


if __name__ == "__main__":
    unittest.main()
