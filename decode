#!/usr/bin/python3

import sys
from scipy.io import wavfile
import numpy as np
from signal_processing_utilities import process_signal
import argparse


def convert_bytes_to_bit_string(data_to_decode, end_zero_padding):
    """This function converts bytes into a string of bits.

    Args:
        data_to_decode (str): This is the string of bytes that will be
                              converted into bits.
        end_zero_padding (int): This is the number of zeroes that were
                              padded into the final byte. They will be
                              removed so the resultant bit string can be
                              properly decompressed.

    Returns:
        bit_string (str): This is the string of bits that will need to
                          be parsed into hexadecimal pairs that
                          represent the bytes of the decompressed wave
                          file.
    """
    bit_string = ""
    for byteIndex in range(0, len(data_to_decode)):
        bit_string_byte = format(data_to_decode[byteIndex], "b")
        bit_string_byte = "0" * (8 - len(bit_string_byte)) + bit_string_byte
        bit_string += bit_string_byte
    bit_string = bit_string[: len(bit_string) - end_zero_padding]
    return bit_string


def find_key_by_value_in_node_mapping_dictionary(
    val_str_to_find, node_mapping_dictionary
):
    """This function allows the searching of the node_mapping_dictionary
    for a key given a value.

    Args:
        val_str_to_find (str): This is expected to be a string of bits.
        node_mapping_dictionary (dict): This is the dictionary that maps
                                      hexadecimal values as keys in the
                                      dictionary to uniquely identifying
                                      strings of bits as values.

    Returns:
        key_mapped_to_value (list): This function returns the
                                    hexadecimal pair found as a key in
                                    the node_mapping_dictionary given a
                                    string of bit values. If the given
                                    string of bits is not found in the
                                    dictionary, the return value is
                                    'None'.
    """
    try:
        key_mapped_to_value = list(node_mapping_dictionary.keys())[
            list(node_mapping_dictionary.values()).index(val_str_to_find)
        ]
    except ValueError:
        return None
    return key_mapped_to_value


def huffman_decoding(huffman_encoded_data: str):
    """This is the algorithm that decodes a huffman encoded string of bytes.

    Args:
        huffman_encoded_data (str): This is the string of bytes to be
                                    decoded.

    Returns:
        decoded_wav_bytes (bytes): This is the byte string that has been
                                   decoded by the huffman decoding
                                   algorithm.
    """

    # Capturing the indices of the huffman_encoded_data
    """ The last two bytes of the huffman_encoded_data are the length of
        the indices within the huffman_encoded_data.
    """
    indices_length = int.from_bytes(huffman_encoded_data[-2:])

    reconstructed_indices_bytes = huffman_encoded_data[-(indices_length) - 2 : -2]

    reconstructed_indices = [
        int.from_bytes(reconstructed_indices_bytes[index : index + 4])
        for index in range(0, len(reconstructed_indices_bytes), 4)
    ]

    # Capturing the End Zero Padding:
    end_zero_padding = reconstructed_indices[-1]

    # Node Mapping Dictionary Keys:
    reconstructed_node_mapping_dictionary_keys_byte_string = huffman_encoded_data[
        0 : reconstructed_indices[0]
    ]

    reconstructed_node_mapping_dictionary_keys_string = str(
        reconstructed_node_mapping_dictionary_keys_byte_string, encoding="utf-8"
    )
    reconstructed_node_mapping_dictionary_keys_l = [
        reconstructed_node_mapping_dictionary_keys_string[index : index + 2]
        for index in range(0, len(reconstructed_node_mapping_dictionary_keys_string), 2)
    ]

    # Node Mapping Dictionary Values Expansion:

    # Node Mapping Dictionary Values (rle_compressed):
    rle_compressed_bytes = huffman_encoded_data[
        reconstructed_indices[0] : reconstructed_indices[1]
    ]

    # Node Mapping Dictionary Values RLE Compressed Indices (rle_compressed):
    rle_locations_compressed_byte_string = huffman_encoded_data[
        reconstructed_indices[1] : reconstructed_indices[2]
    ]

    node_mapping_dictionary_values_byte_string, _ = process_signal.rle_bit_compression(
        byte_string=rle_compressed_bytes,
        rle_locations=rle_locations_compressed_byte_string,
        compress=False,
    )

    # Node Mapping Dictionary Values Indices:
    node_mapping_dict_values_indices_length_compressed_byte_string = (
        huffman_encoded_data[reconstructed_indices[2] : reconstructed_indices[3]]
    )
    reconstructed_node_mapping_dict_values_indices_length_l = process_signal.decode_rle(
        node_mapping_dict_values_indices_length_compressed_byte_string
    )

    # Capturing the bit_string:
    bit_string_bytes = huffman_encoded_data[
        reconstructed_indices[3] : reconstructed_indices[4]
    ]

    bit_string = convert_bytes_to_bit_string(bit_string_bytes, end_zero_padding)

    # Parsing the Node Mapping Dictionary
    byte_string_index = 0

    reconstructed_node_mapping_dictionary = {}

    for index in range(0, len(reconstructed_node_mapping_dictionary_keys_l)):
        # Key
        key_str = reconstructed_node_mapping_dictionary_keys_l[index]

        # Value original dictionary is bytes
        value_byte = node_mapping_dictionary_values_byte_string[
            byte_string_index : byte_string_index
            + reconstructed_node_mapping_dict_values_indices_length_l[index]
        ]
        value_str = str(value_byte).lstrip("b'").rstrip("'")
        reconstructed_node_mapping_dictionary[key_str] = value_str
        byte_string_index += reconstructed_node_mapping_dict_values_indices_length_l[
            index
        ]

    reconstructed_node_mapping_dictionary_sorted = dict(
        sorted(
            reconstructed_node_mapping_dictionary.items(),
            key=lambda items: len(items[1]),
        )
    )

    # Parse the string of bits into hexadecimal values.
    hex_value_array = []
    bitLength = 0
    while len(bit_string) > 0:
        key = find_key_by_value_in_node_mapping_dictionary(
            bit_string[:bitLength], reconstructed_node_mapping_dictionary_sorted
        )
        if key is not None:
            hex_value_array.append(key)
            bit_string = bit_string[bitLength:]
            bitLength = 0
        else:
            bitLength += 1

    hex_wav_str = ""
    hex_wav_str = hex_wav_str.join(hex_value_array)
    decoded_wav_bytes = bytes.fromhex(hex_wav_str)
    return decoded_wav_bytes


def read_encoded_file(compressed_file_path: str):
    """The main driving method that will decode a huffman encoded file.

    Args:
        compressed_file_path (str, optional): The path of the compressed
                                              file to decompress.
    """

    # Retrieve the encoded file for decoding and parse the file.
    with open(compressed_file_path, "rb+") as file:
        huffman_encoded_data = file.read()
    return huffman_encoded_data


def process_huffman_encoded_file(args=None):
    """This is the driver function that processes a huffman encoded file
    format.

    Args:
        args: This is the list of arguments that include the compressed
        and decompressed file paths. These arguments are parsed from the
        command line at runtime.
    """

    huffman_encoded_data = read_encoded_file(
        compressed_file_path=args.compressed_file_path
    )
    decoded_wav_bytes = huffman_decoding(huffman_encoded_data)

    # The sample rate of the data is known in advance.
    wavfile.write(
        filename=args.decompressed_file_path,
        sample_rate=19531,
        data=np.frombuffer(decoded_wav_bytes, dtype=np.int16),
    )


def process_spike_detection_huffman_encoded_data(args=None):
    """This is the driver function that processes a huffman encoded file
    format that has been encoded in such a way as to only detect neural
    spikes.

    Args:
        args: Thes are the parsed command line arguments. These
            arguments contain the compressed and decompressed file
            paths.
    """

    huffman_encoded_data = read_encoded_file(
        compressed_file_path=args.compressed_file_path
    )
    decoded_wav_bytes = huffman_decoding(huffman_encoded_data)
    encoded_data = process_signal.convert_byte_string_to_encoded_data(decoded_wav_bytes)
    sample_rate, amplitude_array = process_signal.decode_data(encoded_data)
    wavfile.write(
        filename=args.decompressed_file_path,
        sample_rate=sample_rate,
        data=amplitude_array,
    )


def decompress(byte_string: bytes):
    """This function accepts a compressed byte string compressed
    using "brainwire" compression from the encode module. It then
    decompresses this data into the original array of amplitudes except
    only detected neural spike information is present. There are
    zero-valued amplitudes at all other locations of the original
    waveform. The decompressed representation returned as the
    sample_rate and corresponding amplitude_array.

    Args:
        file (str): This is the string of the compressed file path. The
                    expected encoding file type is ".brainwire"
    """
    decoded_wav_bytes = huffman_decoding(byte_string)
    encoded_data = process_signal.convert_byte_string_to_encoded_data(
        encoded_data_byte_string=decoded_wav_bytes
    )
    sample_rate, amplitude_array = process_signal.decode_data(encoded_data)
    return sample_rate, amplitude_array


def initialize_argument_parser():
    """This function will initialize the argument parser with command
    line arguments.

    Returns:
        This function will return the parsed arguments from the command
        line.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "compressed_file_path",
        help="This is the compressed output file path. It is presumed to end this new file name with a '.brainwire' file extension. A sample file name is 'compressed_file.wav.brainwire.",
    )
    parser.add_argument(
        "decompressed_file_path",
        help="This is the absolute file path to the reconstructed raw neural data. This is used to name the output file along with the extension. A sample file extension is 'reconstructed_neural_data.wav.brainwire.copy'.",
    )
    parser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="This option will increase compression speed at the cost of compression size by exclusively implementing a huffman-encoding algorithm.",
    )
    args = parser.parse_args()

    print("compressed_file_path: {}".format(args.compressed_file_path))
    print("decompressed_file_path: {}".format(args.decompressed_file_path))
    return args


def main():
    """This is the main driver logic of the decode function."""
    args = initialize_argument_parser()
    if args.quick:
        process_huffman_encoded_file(args=args)
    else:
        process_spike_detection_huffman_encoded_data(args=args)


if __name__ == "__main__":
    main()
