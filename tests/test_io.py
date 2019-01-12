import os
import sys
import math


if sys.version_info >= (3, 0):
    PYTHON_3 = True
else:
    PYTHON_3 = False


def _sample_generator(*data_buffers):
    """
    Takes a list of many mono audio data buffers and makes a sample generator
    of interleaved audio samples, one sample from each channel. The resulting
    generator can be used to build a multichannel audio buffer.
    >>> gen = _sample_generator("abcd", "ABCD")
    >>> list(gen)
    ["a", "A", "b", "B", "c", "C", "d", "D"]
    """
    frame_gen = zip(*data_buffers)
    return (sample for frame in frame_gen for sample in frame)


def _array_to_bytes(a):
    """
    Converts an `array.array` to `bytes`.
    """
    if PYTHON_3:
        return a.tobytes()
    else:
        return a.tostring()
