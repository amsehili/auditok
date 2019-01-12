import os
import sys
import math
from array import array
from auditok.io import DATA_FORMAT


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


def _generate_pure_tone(
    frequency, duration_sec=1, sampling_rate=16000, sample_width=2, volume=1e4
):
    """
    Generates a pure tone with the given frequency.
    """
    assert frequency <= sampling_rate / 2
    max_value = (2 ** (sample_width * 8) // 2) - 1
    if volume > max_value:
        volume = max_value
    fmt = DATA_FORMAT[sample_width]
    total_samples = int(sampling_rate * duration_sec)
    step = frequency / sampling_rate
    two_pi_step = 2 * math.pi * step
    data = array(
        fmt,
        (
            int(math.sin(two_pi_step * i) * volume)
            for i in range(total_samples)
        ),
    )
    return data


PURE_TONE_DICT = {
    freq: _generate_pure_tone(freq, 1, 16000, 2) for freq in (400, 800, 1600)
}
PURE_TONE_DICT.update(
    {
        freq: _generate_pure_tone(freq, 0.1, 16000, 2)
        for freq in (600, 1150, 2400, 7220)
    }
)
