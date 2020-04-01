import unittest
from unittest import TestCase
from unittest.mock import patch
import math
from array import array as array_
from genty import genty, genty_dataset
from auditok.util import (
    AudioEnergyValidator,
    make_duration_formatter,
    make_channel_selector,
)
from auditok import signal as signal_
from auditok import signal_numpy

from auditok.exceptions import TimeFormatError


def _sample_generator(*data_buffers):
    """
    Takes a list of many mono audio data buffers and makes a sample generator
    of interleaved audio samples, one sample from each channel. The resulting
    generator can be used to build a multichannel audio buffer.
    >>> gen = _sample_generator("abcd", "ABCD")
    >>> list(gen)
    ["a", "A", 1, 1, "c", "C", "d", "D"]
    """
    frame_gen = zip(*data_buffers)
    return (sample for frame in frame_gen for sample in frame)


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
    fmt = signal_.FORMAT[sample_width]
    total_samples = int(sampling_rate * duration_sec)
    step = frequency / sampling_rate
    two_pi_step = 2 * math.pi * step
    data = array_(
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
class TestFunctions(TestCase):
    def setUp(self):
        self.data = b"012345679ABC"

    @genty_dataset(
        only_seconds=("%S", 5400, "5400.000"),
        only_millis=("%I", 5400, "5400000"),
        full=("%h:%m:%s.%i", 3725.365, "01:02:05.365"),
        full_zero_hours=("%h:%m:%s.%i", 1925.075, "00:32:05.075"),
        full_zero_minutes=("%h:%m:%s.%i", 3659.075, "01:00:59.075"),
        full_zero_seconds=("%h:%m:%s.%i", 3720.075, "01:02:00.075"),
        full_zero_millis=("%h:%m:%s.%i", 3725, "01:02:05.000"),
        duplicate_directive=(
            "%h %h:%m:%s.%i %s",
            3725.365,
            "01 01:02:05.365 05",
        ),
        no_millis=("%h:%m:%s", 3725, "01:02:05"),
        no_seconds=("%h:%m", 3725, "01:02"),
        no_minutes=("%h", 3725, "01"),
        no_hours=("%m:%s.%i", 3725, "02:05.000"),
    )
    def test_make_duration_formatter(self, fmt, duration, expected):
        formatter = make_duration_formatter(fmt)
        result = formatter(duration)
        self.assertEqual(result, expected)

    @genty_dataset(
        duplicate_only_seconds=("%S %S",),
        duplicate_only_millis=("%I %I",),
        unknown_directive=("%x",),
    )
    def test_make_duration_formatter_error(self, fmt):
        with self.assertRaises(TimeFormatError):
            make_duration_formatter(fmt)

    @genty_dataset(
        int8_1channel_select_0=(
            1,
            1,
            0,
            [48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67],
        ),
        int8_2channel_select_0=(1, 2, 0, [48, 50, 52, 54, 57, 66]),
        int8_3channel_select_0=(1, 3, 0, [48, 51, 54, 65]),
        int8_3channel_select_1=(1, 3, 1, [49, 52, 55, 66]),
        int8_3channel_select_2=(1, 3, 2, [50, 53, 57, 67]),
        int8_4channel_select_0=(1, 4, 0, [48, 52, 57]),
        int16_1channel_select_0=(
            2,
            1,
            0,
            [12592, 13106, 13620, 14134, 16697, 17218],
        ),
        int16_2channel_select_0=(2, 2, 0, [12592, 13620, 16697]),
        int16_2channel_select_1=(2, 2, 1, [13106, 14134, 17218]),
        int16_3channel_select_0=(2, 3, 0, [12592, 14134]),
        int16_3channel_select_1=(2, 3, 1, [13106, 16697]),
        int16_3channel_select_2=(2, 3, 2, [13620, 17218]),
        int32_1channel_select_0=(4, 1, 0, [858927408, 926299444, 1128415545],),
        int32_3channel_select_0=(4, 3, 0, [858927408]),
        int32_3channel_select_1=(4, 3, 1, [926299444]),
        int32_3channel_select_2=(4, 3, 2, [1128415545]),
    )
    def test_make_channel_selector_one_channel(
        self, sample_width, channels, selected, expected
    ):

        # force using signal functions with standard python implementation
        with patch("auditok.util.signal", signal_):
            selector = make_channel_selector(sample_width, channels, selected)
            result = selector(self.data)

        fmt = signal_.FORMAT[sample_width]
        expected = array_(fmt, expected)
        if channels == 1:
            expected = bytes(expected)
        self.assertEqual(result, expected)

        # Use signal functions with numpy implementation
        with patch("auditok.util.signal", signal_numpy):
            selector = make_channel_selector(sample_width, channels, selected)
            resutl_numpy = selector(self.data)

        expected = array_(fmt, expected)
        if channels == 1:
            expected = bytes(expected)
            self.assertEqual(resutl_numpy, expected)
        else:
            self.assertTrue(all(resutl_numpy == expected))


@genty
class TestAudioEnergyValidator(TestCase):
    @genty_dataset(
        mono_valid_uc_None=([350, 400], 1, None, True),
        mono_valid_uc_any=([350, 400], 1, "any", True),
        mono_valid_uc_0=([350, 400], 1, 0, True),
        mono_valid_uc_mix=([350, 400], 1, "mix", True),
        # previous cases are all the same since we have mono audio
        mono_invalid_uc_None=([300, 300], 1, None, False),
        stereo_valid_uc_None=([300, 400, 350, 300], 2, None, True),
        stereo_valid_uc_any=([300, 400, 350, 300], 2, "any", True),
        stereo_valid_uc_mix=([300, 400, 350, 300], 2, "mix", True),
        stereo_valid_uc_avg=([300, 400, 350, 300], 2, "avg", True),
        stereo_valid_uc_average=([300, 400, 300, 300], 2, "average", True),
        stereo_valid_uc_mix_with_null_channel=(
            [634, 0, 634, 0],
            2,
            "mix",
            True,
        ),
        stereo_valid_uc_0=([320, 100, 320, 100], 2, 0, True),
        stereo_valid_uc_1=([100, 320, 100, 320], 2, 1, True),
        stereo_invalid_uc_None=([280, 100, 280, 100], 2, None, False),
        stereo_invalid_uc_any=([280, 100, 280, 100], 2, "any", False),
        stereo_invalid_uc_mix=([400, 200, 400, 200], 2, "mix", False),
        stereo_invalid_uc_0=([300, 400, 300, 400], 2, 0, False),
        stereo_invalid_uc_1=([400, 300, 400, 300], 2, 1, False),
        zeros=([0, 0, 0, 0], 2, None, False),
    )
    def test_audio_energy_validator(
        self, data, channels, use_channel, expected
    ):

        data = array_("h", data)
        sample_width = 2
        energy_threshold = 50
        validator = AudioEnergyValidator(
            energy_threshold, sample_width, channels, use_channel
        )

        if expected:
            self.assertTrue(validator.is_valid(data))
        else:
            self.assertFalse(validator.is_valid(data))


if __name__ == "__main__":
    unittest.main()
