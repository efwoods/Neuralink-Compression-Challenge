# [Neuralink Compression Challenge](https://content.neuralink.com/compression-challenge/README.html)

## Summary

[content.neuralink.com/compression-challenge/data.zip](https://content.neuralink.com/compression-challenge/data.zip) is one hour of raw electrode recordings from a Neuralink implant.

This Neuralink is implanted in the motor cortex of a non-human primate, and recordings were made while playing a video game, [like this](https://www.youtube.com/watch?v=rsCul1sp4hQ).

Compression is essential: N1 implant generates ~200Mbps of eletrode data (1024 electrodes @ 20kHz, 10b resolution) and can transmit ~1Mbps wirelessly.
So > 200x compression is needed.
Compression must run in real time (< 1ms) at low power (< 10mW, including radio).

Neuralink is looking for new approaches to this compression problem, and exceptional engineers to work on it.

## Task

Build executables ./encode and ./decode which pass eval.sh. This verifies compression is lossless and measures compression ratio.

Your submission will be scored on the compression ratio it achieves on a different set of electrode recordings.
Bonus points for optimizing latency and power efficiency

Submit with source code and build script. Should at least build on Linux.

## [Pypi Brainwire Package](https://pypi.org/project/brainwire/)

Encode and Decode are now available under the [brainwire](https://pypi.org/project/brainwire/) pypi package. The modules will detect neural spikes, compress the data, and decompress the data to present a sample rate and array of amplitudes that only contain detected spike information. They can be downloaded using `pip install brainwire` and implemented with `brainwire.encode.compress(file_path)` or `brainwire.decode.decompress(compressed_brainwire_data_format)` where 'compressed_brainwire_data_format' is the compressed data returned from 'brainwire.encode.compress()' and the 'file_path' is the path to raw neural data in '.wav' format.

## Sample Compression Results:

### Encoding Time:

- Time Δ Microseconds: 3686937.0 μs
- Time Δ Milliseconds: 3686.937 ms
- Time Δ Seconds: 3.686937 s

### Decoding Time:

- Time Δ Microseconds: 3871124.0 μs
- Time Δ Milliseconds: 3871.124 ms
- Time Δ Seconds: 3.871124 s

### Total Time (encode and decode):

- Time Δ Microseconds: 7558100.0 μs
- Time Δ Milliseconds: 7558.1 ms
- Time Δ Seconds: 7.5581 s

### Compression Ratio

- Original File Size: 197526
- Compressed File Size: 92251
- Percent of Compression: 53.30%
