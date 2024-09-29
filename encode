#!/usr/bin/python3

import heapq
from scipy.io import wavfile
import numpy as np
from signal_processing_utilities import process_signal
import argparse


class Node:
    """Purpose: The Node class is used to sort the freqencies of the
    hexadecimal values into a binary tree. The binary tree is
    then used to identify a binary mapping that uniquely
    represents each hexadecimal byte-pair. This is the atomic
    unit of the Huffman encoding technique.
    """

    def __init__(self, freq, data, left=None, right=None):
        self.freq = freq
        self.data = data
        self.left = left
        self.right = right
        self.code = ""

    def __lt__(self, nxt):
        return self.freq < nxt.freq


def create_node_mapping_dictionary(node, val="", node_mapping_dict={}):
    """Creates a mapping of unique binary strings to unique bytes of
           pairs of hexidecimal values found within the input wave file.

    Args:
        node (Node): This is a node of class Node. It is used to build
                     the binary tree.
        val (str, optional): This is the huffman code that is being
                             built into the unique representation of the
                             hexadecimal pair. Defaults to ''.
        node_mapping_dict (dict, optional): This is the dictionary that
                                          contains the mapping of
                                          hexadecimal values to unique
                                          binary string representations.
                                          Defaults to {}.

    Returns:
        bytes: This function returns the bytes of the wave file that was
               read.
    """
    # Is this value a string or a
    newVal = val + str(node.code)
    # if node is not an edge node, traverse
    if node.left:
        create_node_mapping_dictionary(node.left, newVal, node_mapping_dict)
    if node.right:
        create_node_mapping_dictionary(node.right, newVal, node_mapping_dict)
    if not node.left and not node.right:
        node_mapping_dict[node.data] = newVal
        # print(f"{node.data} -> {newVal}")
    return node_mapping_dict


def determine_hex_freq(input_wav):
    """This function determines how frequent a hexadecimal pair of
        values occurs within a string of bytes.

    Args:
        input_wav (bytes): This is the string of bytes of the wav file
                           that was read into memory.

    Returns:
        sorted_hex_freq_dictionary: This function returns a dictionary
                                    of hexpairs which are mapped to
                                    frequency of occurence in the input
                                    wave file.
    """
    input_wav_hex = input_wav.hex()
    hex_freq_dictionary = {}
    for digit_position in range(0, len(input_wav.hex()), 2):
        current_hex_pair = (
            input_wav_hex[digit_position] + input_wav_hex[digit_position + 1]
        )
        try:
            hex_freq_dictionary[current_hex_pair] += 1
        except KeyError:
            hex_freq_dictionary[current_hex_pair] = 1
    sorted_hex_freq_dictionary = dict(
        sorted(hex_freq_dictionary.items(), key=lambda x: x[1])
    )
    return sorted_hex_freq_dictionary


def convertHexToBit(input_wav, node_mapping_dict):
    """This function uses a dictionary of node mappings to convert
          the hexadecimal representation of the of the input wave file
          to a string of bits. This representation is a compressed form
          of the hexadecimal representation of the input wave file.

    Args:
        input_wav (bytes): This is the path of the file to be read into
                           memory.
        node_mapping_dict (dict): This is the dictionary mapping bits to
                                the hexadecimal representation of the
                                input wav file.

    Returns:
        bit_string: This is the string of bits that will be written to
                   the output file.
        lenend_zero_padding: This is the number of zeroes that have been
                             padded onto the end of the bit string to
                             make a complete set of bytes.
    """
    hex_input_wav = input_wav.hex()
    bit_string = ""
    for index in range(0, len(hex_input_wav), 2):
        hex_pair = hex_input_wav[index] + hex_input_wav[index + 1]
        bit_string += node_mapping_dict[hex_pair]
    end_zero_padding = "0" * (8 - (len(bit_string) % 8))
    bit_string += end_zero_padding
    len_end_zero_padding = len(end_zero_padding)
    return bit_string, len_end_zero_padding


def create_byte_string(
    node_mapping_dict,
    bit_string,
    end_zero_padding,
):
    """This function writes the encoded outputfile.

    Args:
        node_mapping_dict (dict): This is the dictionary which maps each
                                  hexadecimal representation of the
                                  bytes of the input wave file to a
                                  string of bits whose length is
                                  dependent on frequency of the
                                  hexadecimal pair.
        bit_string (str): This is the string of bits that is
                            interpreted from the original input wave
                            file and parsed using the node_mapping_dict.
        end_zero_padding (str): This is the number of zeroes that are
                                 padded to the end of the bit_string to
                                 create a full set of bytes to be
                                 written to the output data file.
    Returns:
        byte_string (bytes): This function returns a bytes object of the
                             compressed data.

    Notes:
        The encoding is the node_mapping_dictionary keys written as
        bytes, the run-length-encoded node_mapping_dictionary values
        written as bytes, the run-length-encoded lengths of the rle
        node_mapping_dictionary values written as bytes, the
        run-length-encoding of the node_mapping_dictionary value
        lengths, huffman encoded data as bytes, the indices of the start
        and end of each data portion as mentioned here, and the size in
        bytes of the indices. The last index of the indices array
        contains the value of the number of zeros that the encoded data
        is padded with. This is because the last index would be the
        length of the encoded data file, but this information can be
        interpreted otherwise.
    """
    num_bytes = 1
    byteorder = "big"
    indices = []
    byte_string = b""
    node_mapping_dict_keys_list = list(node_mapping_dict.keys())
    node_mapping_dict_values_list = list(node_mapping_dict.values())

    # Node Mapping Dictionary Keys appended to byte_string:
    node_mapping_dict_keys_list_byte_string_l = [
        bytes(node_mapping_dict_keys_list[index], encoding="utf-8")
        for index in range(len(node_mapping_dict_keys_list))
    ]

    for key_value in node_mapping_dict_keys_list_byte_string_l:
        byte_string += key_value

    # Node Mapping Dictionary Keys: indices[0]
    indices.append(len(byte_string))

    # Node Mapping Dictionary Values appended to byte_string:
    node_mapping_dict_values_list_byte_string_l = [
        bytes(node_mapping_dict_values_list[index], encoding="utf-8")
        for index in range(len(node_mapping_dict_values_list))
    ]

    node_mapping_dict_values_byte_string = b""
    for value_bytes in node_mapping_dict_values_list_byte_string_l:
        node_mapping_dict_values_byte_string += value_bytes

    (
        rle_compressed_node_mapping_dictionary_values_bytes,
        rle_locations_compressed_byte_string,
    ) = process_signal.rle_bit_compression(
        byte_string=node_mapping_dict_values_byte_string
    )

    byte_string += rle_compressed_node_mapping_dictionary_values_bytes

    # Node Mapping Dictionary Values (rle_compressed): indices[1]
    indices.append(len(byte_string))

    # Node Mapping Dictionary Values RLE Compressed Indices (rle_compressed): indices[2]
    byte_string += rle_locations_compressed_byte_string
    indices.append(len(byte_string))

    # Run-Length-Encoding the node mapping dictionary value lengths
    node_mapping_dict_values_indices_length_list = [
        len(node_mapping_dict_values_list[index])
        for index in range(len(node_mapping_dict_values_list))
    ]
    # This information is non-binary ∴ encode_rle is used as encoding.
    # node_mapping_dict_values_indices_length_list_compresed has a known
    # format index pair of "value, frequency" for each "index, index+1"
    # within the array.
    node_mapping_dict_values_indices_length_list_compresed, rle_locations = (
        process_signal.encode_rle(node_mapping_dict_values_indices_length_list)
    )

    node_mapping_dict_values_indices_length_compressed_byte_string_l = [
        node_mapping_dict_values_indices_length_list_compresed[index].to_bytes(
            1, byteorder="big"
        )
        for index in range(len(node_mapping_dict_values_indices_length_list_compresed))
    ]

    for byte in node_mapping_dict_values_indices_length_compressed_byte_string_l:
        byte_string += byte

    # Node Mapping Dictionary Values Indices
    #   (rle_compressed via encode_rle): indices[3]
    indices.append(len(byte_string))

    for index in range(0, len(bit_string), 8):
        byte_to_write = bit_string[index : index + 8]
        int_of_byte_to_write = int(byte_to_write, base=2)
        byte_string += int_of_byte_to_write.to_bytes(num_bytes, byteorder)

    # This is the length of the huffman encoded string of bits stored as
    # rle bytes: indices[4]
    indices.append(len(byte_string))

    # This is the number of zeros that have been padded to the ultimate
    # byte_string index in order to make it equal to a byte.
    # end_zero_padding: indices[5]
    indices.append(end_zero_padding)

    bytes_indices_list = [index.to_bytes(4, "big") for index in indices]
    for byte in bytes_indices_list:
        byte_string += byte
    bytes_indices_size = 4 * len(bytes_indices_list)

    # The line of code below is presuming the
    # "bytes_indices_size" is < 256
    byte_string += bytes_indices_size.to_bytes(2, byteorder=byteorder)
    return byte_string


def huffman_encoding(
    input_data: np.ndarray,
):
    """Main method to drive the encoding operation implementing huffman
    encoding.

    Args:
        input_data (np.ndarray): This is the input wave file
                                           as a numpy array. If this
                                           input is given, the data will
                                           be converted to bytes.

    Returns:
        node_mapping_dict (dict): This is the dictionary containing
                                hexadecimal keys and their corresponding
                                binary string values.
        bit_string (str): This is the string of bits that represent the
                         huffman encoded data.
        end_zero_padding (int): This is the integer that indicates how
                              many zeros have been added to the bit
                              string to make the length evenly divisible
                              by eight such that each eight bits may be
                              converted into individual bytes.
    """

    sorted_hex_freq_dict = determine_hex_freq(
        input_data if type(input_data) == bytes else input_data.tobytes()
    )
    hex_freq_values = list(sorted_hex_freq_dict.values())
    hex_freq_keys = list(sorted_hex_freq_dict.keys())

    # Create a list of nodes
    nodes = []
    for item in range(len(hex_freq_keys)):
        heapq.heappush(nodes, Node(hex_freq_values[item], hex_freq_keys[item]))

    # Build the node tree
    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        left.code = 0
        right.code = 1
        newNode = Node(
            left.freq + right.freq, left.data + right.data, left=left, right=right
        )
        heapq.heappush(nodes, newNode)
    node_mapping_dict = create_node_mapping_dictionary(
        nodes[0], val="", node_mapping_dict={}
    )

    # The hexadecimal representation of the bytes of the input wave file
    # is converted to a string of bits to write
    bit_string, end_zero_padding = convertHexToBit(
        input_data if type(input_data) == bytes else input_data.tobytes(),
        node_mapping_dict,
    )

    return node_mapping_dict, bit_string, end_zero_padding


def create_huffman_encoded_file(args=None):
    """This driver function will read the data before huffman encoding
    the data and writing the resulting string of bytes to a file.

    Args:
        args: These are the arguments parsed from the command line.
                If defined, they are expected to contain the 'file_path'
                of the raw neural data to be compressed and the
                'compressed_file_path' that will be the output return
                value after encoding the original '.wav' file. Defaults
                to None.
    """

    sample_rate, input_wav = wavfile.read(filename=args.file_path)
    node_mapping_dict, bit_string, end_zero_padding = huffman_encoding(
        input_data=input_wav
    )
    byte_string = create_byte_string(node_mapping_dict, bit_string, end_zero_padding)
    process_signal.write_file_bytes(
        file_path=args.compressed_file_path, data_bytes=byte_string
    )


def implement_spike_detection_module_and_huffman_encode_file(args=None):
    """This driver function will read data, preprocess the data, detect neural
    spikes, create an object containing only the detected spikes,
    convert this object to a string of bytes, huffman encode those
    bytes, and write the huffman encoded representation of the bytes to
    a file.

    Args:
        args: These are the arguments parsed from the command line.
                If used, they are expected to contain the 'file_path'
                of the raw neural data to be compressed and the
                'compressed_file_path' that will be the output return
                value after encoding the original '.wav' file. Defaults
                to None.
    """

    # Read Data
    sample_rate, input_wav = wavfile.read(file=args.file_path)

    # Preprocess Data & Detect Spikes
    filtered_data_bandpass = process_signal.preprocess_signal(
        raw_neural_signal=input_wav, sample_rate=sample_rate
    )
    spike_train_time_index_list = process_signal.detect_neural_spikes(
        filtered_data_bandpass
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

    byte_string = huffman_encoding(
        input_data=encoded_data_byte_string,
    )

    process_signal.write_file_bytes(
        file_path=args.compressed_file_path, data_bytes=byte_string
    )


def compress(file: str):
    """This function accepts a file path to compress. It will read data,
    preprocess the data, detect neural spikes, create an object
    containing only the detected spikes, convert this object to a string
    of bytes, huffman encode those bytes, and will return a bytes object
    containing the compressed data.

    Args:
        file (str): This is the path to the file to be compressed.
    """
    sample_rate, input_wav = wavfile.read(file)
    filtered_data_bandpass = process_signal.preprocess_signal(
        raw_neural_signal=input_wav, sample_rate=sample_rate
    )
    spike_train_time_index_list = process_signal.detect_neural_spikes(
        filtered_data_bandpass
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
    node_mapping_dict, bit_string, end_zero_padding = huffman_encoding(
        input_data=encoded_data_byte_string
    )

    byte_string = create_byte_string(node_mapping_dict, bit_string, end_zero_padding)
    return byte_string


def initialize_argument_parser():
    """This function will initialize the argument parser with command
    line arguments.

    Returns:
        This function will return the parsed arguments from the command
        line.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path",
        help="This is the absolute file path to the raw neural data with a '.wav' file extension.",
    )
    parser.add_argument(
        "compressed_file_path",
        help="This is the compressed output file path. It is presumed to end this new file name with a '.brainwire' file extension.",
    )
    parser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="This option will increase compression speed at the cost of compression size by exclusively implementing a huffman-encoding algorithm.",
    )
    args = parser.parse_args()

    print("file: {}".format(args.file_path))
    print("compressed_file_path: {}".format(args.compressed_file_path))
    return args


def main():
    """This is the main driver of the code."""
    args = initialize_argument_parser()
    if args.quick:
        create_huffman_encoded_file(args=args)
    else:
        implement_spike_detection_module_and_huffman_encode_file(args=args)


if __name__ == "__main__":
    main()
