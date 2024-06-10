"""This module is used to test the encode module."""

import unittest
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
import logging

import wave

import scipy.signal

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

    def test04_read_wave_information(self):
        """This is a test that the information of the wave file is read.
        """

        input_wav = wave.open(self.file, 'rb')
        logging.info("input_wav type: {}".format(type(input_wav)))
        logging.info("Channels: {}".format(input_wav.getnchannels()))
        logging.info("Sample width {} Bytes".format(input_wav.getsampwidth()))
        logging.info("Frequency: {}".format(input_wav.getframerate(), "kHz"))
        logging.info("Number of frames: {}".format(input_wav.getnframes()))
        logging.info("Audio length: {:.2f} seconds".format(input_wav.getnframes() / 
                                               input_wav.getframerate()))
        pred_num_bytes = input_wav.getnframes() * input_wav.getnchannels() \
            * input_wav.getsampwidth()

        sample_bytes = input_wav.readframes(input_wav.getnframes())
        self.assertEqual(pred_num_bytes, len(sample_bytes))
        logging.info(f"pred_num_bytes: {pred_num_bytes}")
        logging.info(f"len(sample_bytes): {len(sample_bytes)}")

if __name__ == '__main__':
    unittest.main()
