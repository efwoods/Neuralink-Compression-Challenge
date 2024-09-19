# [Neuralink Compression Challenge](https://content.neuralink.com/compression-challenge/README.html)

## Summary

content.neuralink.com/compression-challenge/data.zip is one hour of raw electrode recordings from a Neuralink implant.

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
