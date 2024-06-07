"""This module is used to test the encode module."""

import unittest
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
import logging

# Log all messages from all logging levels
logging.basicConfig(level = logging.DEBUG)

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
        self.file = 'data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav'
        self.compressed_file_path = 'data/0ab237b7-fb12-4687-afed-8d1e2070d6' \
            '21.wav.brainwire'

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
        """Used to test the read_file method in the encode module.
        """

        logging.info("test_read_input_wav_is_type_bytes")
        input_wav = encode.read_file(self.file)
        self.assertEqual(type(input_wav), bytes)

    def test03_huffman_encoding(self):
        """This is a test of all of the huffman encoding algorithm
        """

        logging.info("test_all")
        encode.huffman_encoding(self.file, self.compressed_file_path)

if __name__ == '__main__':
    unittest.main()
