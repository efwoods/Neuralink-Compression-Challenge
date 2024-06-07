"""This module tests the decoding module named decode."""

import unittest
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
import logging

# Set logging to all logging levels
logging.basicConfig(level=logging.DEBUG)

# Custom import of python file "decode"
spec = spec_from_loader('decode', SourceFileLoader('decode', './decode'))
decode = module_from_spec(spec)
spec.loader.exec_module(decode)

class TestDecode(unittest.TestCase):
    """This class contains test cases for the decode module.

    Args:
        unittest (module): This module enables custom tests.
    """

    def setUp(self):
        self.compressed_file_path = 'data/0ab237b7-fb12-4687-afed-8d1e2070d6' \
            '21.wav.brainwire'
        self.decompressed_file_path = 'data/0ab237b7-fb12-4687-afed-8d1e2070' \
            'd621.wav.copy'

    def tearDown(self) -> None:
        pass

    def test_huffman_decoding(self):
        decode.huffman_decoding(self.compressed_file_path, 
                                self.decompressed_file_path)

if __name__ == '__main__':
    unittest.main()
