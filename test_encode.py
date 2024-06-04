import unittest
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader

spec = spec_from_loader("encode", SourceFileLoader("encode", "./encode"))
encode = module_from_spec(spec)
spec.loader.exec_module(encode)

class TestEncode(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_read_input_wav_is_type_bytes(self):
        filePath = './data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav'
        input_wav = encode.read_file(filePath)
        self.assertEqual(type(input_wav), bytes)
    
if __name__ == '__main__':
    unittest.main()