import os
import sys
import math
from array import array
from tempfile import NamedTemporaryFile
import filecmp
from unittest import TestCase
from genty import genty, genty_dataset
from auditok.io import (
    AudioIOError,
    DATA_FORMAT,
    AudioParameterError,
    check_audio_data,
    _array_to_bytes,
    _mix_audio_channels,
    _extract_selected_channel,
    from_file,
    _save_raw,
    _save_wave,
)


if sys.version_info >= (3, 0):
    PYTHON_3 = True
    from unittest.mock import patch
else:
    PYTHON_3 = False
    from mock import patch

AUDIO_PARAMS_SHORT = {"sr": 16000, "sw": 2, "ch": 1}


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
        mono_1byte=([400], 1),
        stereo_1byte=([400, 600], 1),
        three_channel_1byte=([400, 600, 2400], 1),
        mono_2byte=([400], 2),
        stereo_2byte=([400, 600], 2),
        three_channel_2byte=([400, 600, 1150], 2),
        mono_4byte=([400], 4),
        stereo_4byte=([400, 600], 4),
        four_channel_2byte=([400, 600, 1150, 7220], 4),
    )
    def test_mix_audio_channels(self, frequencies, sample_width):
        sampling_rate = 16000
        sample_width = 2
        channels = len(frequencies)
        mono_channels = [
            _generate_pure_tone(
                freq,
                duration_sec=0.1,
                sampling_rate=sampling_rate,
                sample_width=sample_width,
            )
            for freq in frequencies
        ]
        fmt = DATA_FORMAT[sample_width]
        expected = _array_to_bytes(
            array(
                fmt,
                (sum(samples) // channels for samples in zip(*mono_channels)),
            )
        )
        data = _array_to_bytes(array(fmt, _sample_generator(*mono_channels)))
        mixed = _mix_audio_channels(data, channels, sample_width)
        self.assertEqual(mixed, expected)

    @genty_dataset(
        mono_1byte=([400], 1, 0),
        stereo_1byte_2st_channel=([400, 600], 1, 1),
        mono_2byte=([400], 2, 0),
        stereo_2byte_1st_channel=([400, 600], 2, 0),
        stereo_2byte_2nd_channel=([400, 600], 2, 1),
        three_channel_2byte_last_negative_idx=([400, 600, 1150], 2, -1),
        three_channel_2byte_2nd_negative_idx=([400, 600, 1150], 2, -2),
        three_channel_2byte_1st_negative_idx=([400, 600, 1150], 2, -3),
        three_channel_4byte_1st=([400, 600, 1150], 4, 0),
        three_channel_4byte_last_negative_idx=([400, 600, 1150], 4, -1),
    )
    def test_extract_selected_channel(
        self, frequencies, sample_width, use_channel
    ):

        mono_channels = [
            _generate_pure_tone(
                freq,
                duration_sec=0.1,
                sampling_rate=16000,
                sample_width=sample_width,
            )
            for freq in frequencies
        ]
        channels = len(frequencies)
        fmt = DATA_FORMAT[sample_width]
        expected = _array_to_bytes(mono_channels[use_channel])
        data = _array_to_bytes(array(fmt, _sample_generator(*mono_channels)))
        selected_channel = _extract_selected_channel(
            data, channels, sample_width, use_channel
        )
        self.assertEqual(selected_channel, expected)

    @genty_dataset(
        raw_with_audio_format=(
            "audio",
            "raw",
            "_load_raw",
            AUDIO_PARAMS_SHORT,
        ),
        raw_with_extension=(
            "audio.raw",
            None,
            "_load_raw",
            AUDIO_PARAMS_SHORT,
        ),
        wave_with_audio_format=("audio", "wave", "_load_wave"),
        wav_with_audio_format=("audio", "wave", "_load_wave"),
        wav_with_extension=("audio.wav", None, "_load_wave"),
        format_and_extension_both_given=("audio.dat", "wav", "_load_wave"),
        format_and_extension_both_given_b=("audio.raw", "wave", "_load_wave"),
        no_format_nor_extension=("audio", None, "_load_with_pydub"),
        other_formats_ogg=("audio.ogg", None, "_load_with_pydub"),
        other_formats_webm=("audio", "webm", "_load_with_pydub"),
    )
    def test_from_file(
        self, filename, audio_format, funtion_name, kwargs=None
    ):
        funtion_name = "auditok.io." + funtion_name
        if kwargs is None:
            kwargs = {}
        with patch(funtion_name) as patch_function:
            from_file(filename, audio_format, **kwargs)
        self.assertTrue(patch_function.called)

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

    def test_from_file_no_pydub(self):
        with patch("auditok.io._WITH_PYDUB", False):
            with self.assertRaises(AudioIOError):
                from_file("audio", "mp3")

    @genty_dataset(
        raw_first_channel=("raw", 0, 400),
        raw_second_channel=("raw", 1, 800),
        raw_third_channel=("raw", 2, 1600),
        raw_left_channel=("raw", "left", 400),
        raw_right_channel=("raw", "right", 800),
        wav_first_channel=("wav", 0, 400),
        wav_second_channel=("wav", 1, 800),
        wav_third_channel=("wav", 2, 1600),
        wav_left_channel=("wav", "left", 400),
        wav_right_channel=("wav", "right", 800),
    )
    def test_from_file_multichannel_audio(
        self, audio_format, use_channel, frequency
    ):
        expected = PURE_TONE_DICT[frequency]
        filename = "tests/data/test_16KHZ_3channel_400-800-1600Hz.{}".format(
            audio_format
        )
        sample_width = 2
        audio_source = from_file(
            filename,
            sampling_rate=16000,
            sample_width=sample_width,
            channels=3,
            use_channel=use_channel,
        )
        fmt = DATA_FORMAT[sample_width]
        data = array(fmt, audio_source._buffer)
        self.assertEqual(data, expected)

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
