import os
import sys
import math
from array import array
from tempfile import NamedTemporaryFile
import filecmp
from unittest import TestCase
from genty import genty, genty_dataset
from auditok.io import (
    DATA_FORMAT,
    AudioParameterError,
    check_audio_data,
    _save_raw,
    _save_wave,
)


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


@genty
class TestIO(TestCase):
    @genty_dataset(
        valid_mono=(b"\0" * 113, 1, 1),
        valid_stereo=(b"\0" * 160, 1, 2),
        invalid_mono_sw_2=(b"\0" * 113, 2, 1, False),
        invalid_stereo_sw_1=(b"\0" * 113, 1, 2, False),
        invalid_stereo_sw_2=(b"\0" * 158, 2, 2, False),
    )
    def test_check_audio_data(self, data, sample_width, channels, valid=True):

        if not valid:
            with self.assertRaises(AudioParameterError):
                check_audio_data(data, sample_width, channels)
        else:
            self.assertIsNone(check_audio_data(data, sample_width, channels))

    @genty_dataset(
        mono=("mono_400Hz.raw", (400,)),
        three_channel=("3channel_400-800-1600Hz.raw", (400, 800, 1600)),
    )
    def test_save_raw(self, filename, frequencies):
        filename = "tests/data/test_16KHZ_{}".format(filename)
        sample_width = 2
        fmt = DATA_FORMAT[sample_width]
        mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
        data = _array_to_bytes(array(fmt, _sample_generator(*mono_channels)))
        tmpfile = NamedTemporaryFile()
        _save_raw(tmpfile.name, data)
        self.assertTrue(filecmp.cmp(tmpfile.name, filename, shallow=False))

    @genty_dataset(
        mono=("mono_400Hz.wav", (400,)),
        three_channel=("3channel_400-800-1600Hz.wav", (400, 800, 1600)),
    )
    def test_save_wave(self, filename, frequencies):
        filename = "tests/data/test_16KHZ_{}".format(filename)
        sampling_rate = 16000
        sample_width = 2
        channels = len(frequencies)
        fmt = DATA_FORMAT[sample_width]
        mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
        data = _array_to_bytes(array(fmt, _sample_generator(*mono_channels)))
        tmpfile = NamedTemporaryFile()
        _save_wave(tmpfile.name, data, sampling_rate, sample_width, channels)
        self.assertTrue(filecmp.cmp(tmpfile.name, filename, shallow=False))
