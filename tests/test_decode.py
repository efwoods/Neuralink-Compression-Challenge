"""This module tests the decoding module named decode."""

import unittest
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
import logging
import numpy as np
import time
from signal_processing_utilities import process_signal

# Set logging to all logging levels
logging.basicConfig(level=logging.DEBUG)

# Custom import of python file "decode"
spec = spec_from_loader("decode", SourceFileLoader("decode", "./decode"))
decode = module_from_spec(spec)
spec.loader.exec_module(decode)

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

    @unittest.skip("Debugging using test05")
    def test01_huffman_decoding(self):
        logging.info("This is a test of decoding a huffman encoded file exclusively")
        huffman_encoded_data = decode.read_encoded_file(
            compressed_file_path=self.compressed_file_path,
        )
        decoded_wav_bytes = decode.huffman_decoding(huffman_encoded_data)

    @unittest.skip("Debugging using test05")
    def test02_huffman_decoding_to_encoded_format(self):
        logging.info(
            "This is a test to decode the huffman encoded byte string,"
            + " convert the byte string into the encoded format, and"
            + " reconstruct the amplitude array."
        )

        huffman_encoded_data = decode.read_encoded_file(
            compressed_file_path=self.compressed_file_path,
        )
        decoded_wav_bytes = decode.huffman_decoding(huffman_encoded_data)
        encoded_data = process_signal.convert_byte_string_to_encoded_data(
            encoded_data_byte_string=decoded_wav_bytes
        )
        sample_rate, amplitude_array = process_signal.decode_data(encoded_data)
        decode.write_decoded_wav(
            sample_rate=sample_rate,
            decoded_wav=amplitude_array,
            decompressed_file_path=self.decompressed_file_path,
        )

    @unittest.skip("Debugging using test05")
    def test03_decoding_encoded_byte_string(self):
        logging.info(
            "This test encodes a file using huffman encoding and decodes using huffman encoding."
        )
        # Test 06 from test_encode.py of Huffman Encoding
        encoding_start_time = time.time_ns()
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
        )
        node_mapping_dict, bit_string, end_zero_padding = encode.huffman_encoding(
            input_data=input_wav, compressed_file_path=self.compressed_file_path
        )

        byte_string = encode.create_byte_string(
            node_mapping_dict, bit_string, end_zero_padding
        )
        process_signal.write_file_bytes(
            file_path=compressed_file_path, data_bytes=byte_string
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
        decoded_wav_bytes = decode.huffman_decoding(encoded_data_byte_string)

        amplitude_array = np.frombuffer(decoded_wav_bytes, dtype=np.int16)
        decode.write_decoded_wav(
            sample_rate=19531,
            decoded_wav=amplitude_array,
            decompressed_file_path=self.decompressed_file_path,
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

    @unittest.skip("Debugging using test05")
    def test04_huffman_decode_operates(self):
        logging.info("Testing Huffman Decoding exclusively.")
        huffman_encoded_string = decode.read_encoded_file(
            compressed_file_path=self.compressed_file_path
        )
        decoded_wav_bytes = decode.huffman_decoding(
            huffman_encoded_data=huffman_encoded_string
        )
        decode.write_decoded_wav(
            sample_rate=self.sample_rate,
            decoded_wav=decoded_wav_bytes,
            decompressed_file_path=self.decompressed_file_path,
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

        logging.info(
            "ValueError: setting an array element with a"
            + "sequence. The requested array has an "
            + "inhomogeneous shape after 1 dimensions. The "
            + "detected shape was (3850,) + inhomogeneous "
            + "part. Traceback (most recent call last)"
        )

        # Spike Detection & Huffman Encoding
        encoding_start_time = time.time_ns()

        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
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
            compressed_file_path=self.compressed_file_path,
        )

        byte_string = encode.create_byte_string(
            node_mapping_dict, bit_string, end_zero_padding
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
        decoded_wav_bytes = decode.huffman_decoding(huffman_encoded_data)
        encoded_data = process_signal.convert_byte_string_to_encoded_data(
            encoded_data_byte_string=decoded_wav_bytes
        )
        sample_rate, amplitude_array = process_signal.decode_data(encoded_data)
        decode.write_decoded_wav(
            sample_rate=sample_rate,
            decoded_wav=amplitude_array,
            decompressed_file_path=self.decompressed_file_path,
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

    @unittest.skip("Testing End-to-end functionality in test05")
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
        sample_rate, input_wav, compressed_file_path = encode.read_file(
            self.file, self.compressed_file_path
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
            compressed_file_path=self.compressed_file_path,
        )
        encoding_stop_time = time.time_ns()

        byte_string = encode.create_byte_string(
            node_mapping_dict, bit_string, end_zero_padding
        )

        process_signal.write_file_bytes(
            file_path=self.debug_compressed_file_path, data_bytes=byte_string
        )

        # Decoding
        decoding_start_time = time.time_ns()
        huffman_encoded_data = decode.read_encoded_file(
            compressed_file_path=self.compressed_file_path,
        )
        decoded_wav_bytes = decode.huffman_decoding(huffman_encoded_data)
        encoded_data = process_signal.convert_byte_string_to_encoded_data(
            encoded_data_byte_string=decoded_wav_bytes
        )
        sample_rate, amplitude_array = process_signal.decode_data(encoded_data)
        decode.write_decoded_wav(
            sample_rate=sample_rate,
            decoded_wav=amplitude_array,
            decompressed_file_path=self.decompressed_file_path,
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


if __name__ == "__main__":
    unittest.main()
