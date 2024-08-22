"""This module tests the decoding module named decode."""

import unittest
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
from utility import signal_process

# Set logging to all logging levels
logging.basicConfig(level=logging.DEBUG)

# Custom import of python file "decode"
spec = spec_from_loader("decode", SourceFileLoader("decode", "./decode"))
decode = module_from_spec(spec)
spec.loader.exec_module(decode)


class TestDecode(unittest.TestCase):
    """This class contains test cases for the decode module.

    Args:
        unittest (module): This module enables custom tests.
    """

    def setUp(self):
        self.compressed_file_path = (
            "data/0ab237b7-fb12-4687-afed-8d1e2070d6" "21.wav.brainwire"
        )
        self.decompressed_file_path = (
            "data/0ab237b7-fb12-4687-afed-8d1e2070" "d621.wav.copy"
        )

    def tearDown(self) -> None:
        pass

    @unittest.skip("Testing Huffman Decoding then Decoding Encoded Representation")
    def test01_huffman_decoding(self):
        decode.huffman_decoding(self.compressed_file_path, self.decompressed_file_path)

    def test02_huffman_decoding_to_encoded_format(self):
        logging.info(
            "This is a test to decode the huffman encoded byte string, convert the byte string into the encoded format, and reconstruct the amplitude array."
        )

        huffman_encoded_data = decode.read_huffman_encoded_file(
            compressed_file_path=self.compressed_file_path,
            decompressed_file_path=self.decompressed_file_path,
        )
        decoded_wav_bytes = decode.huffman_decoding(huffman_encoded_data)
        encoded_data = signal_process.convert_byte_string_to_encoded_data(
            encoded_data_byte_string=decoded_wav_bytes
        )
        sample_rate, amplitude_array = signal_process.decode_data(encoded_data)
        decode.write_decoded_wav(
            decoded_wav=amplitude_array,
            decompressed_file_path=self.decompressed_file_path,
        )


if __name__ == "__main__":
    unittest.main()
