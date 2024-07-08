# Options to increase the compression ratio:
1. Examine the data; if duplicated, reproduce.
2. XOR the data; in the case of frames, if the data is different, the data will be replaced by a 1 otherwise, it will be zero. This will create long RLE where the frame did not change, and only the differences need to be noted.