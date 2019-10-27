import os
import sys
import math
from array import array
from tempfile import NamedTemporaryFile, TemporaryDirectory
import filecmp
import unittest
from unittest import TestCase
from unittest.mock import patch, Mock
from genty import genty, genty_dataset
from test_util import _sample_generator, _generate_pure_tone, PURE_TONE_DICT
from auditok.signal import FORMAT
from auditok.io import (
    AudioIOError,
    AudioParameterError,
    BufferAudioSource,
    RawAudioSource,
    WaveAudioSource,
    StdinAudioSource,
    check_audio_data,
    _guess_audio_format,
    _get_audio_parameters,
    _load_raw,
    _load_wave,
    _load_with_pydub,
    get_audio_source,
    from_file,
    _save_raw,
    _save_wave,
    _save_with_pydub,
    to_file,
)

AUDIO_PARAMS_SHORT = {"sr": 16000, "sw": 2, "ch": 1}


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
        extention_and_format_same=("wav", "filename.wav", "wav"),
        extention_and_format_different=("wav", "filename.mp3", "wav"),
        extention_no_format=(None, "filename.wav", "wav"),
        format_no_extension=("wav", "filename", "wav"),
        no_format_no_extension=(None, "filename", None),
        wave_as_wav=("wave", "filename", "wav"),
        wave_as_wav_extension=(None, "filename.wave", "wav"),
    )
    def test_guess_audio_format(self, fmt, filename, expected):
        result = _guess_audio_format(fmt, filename)
        self.assertEqual(result, expected)

    def test_get_audio_parameters_short_params(self):
        expected = (8000, 2, 1)
        params = dict(zip(("sr", "sw", "ch"), expected))
        result = _get_audio_parameters(params)
        self.assertEqual(result, expected)

    def test_get_audio_parameters_long_params(self):
        expected = (8000, 2, 1)
        params = dict(
            zip(
                ("sampling_rate", "sample_width", "channels", "use_channel"),
                expected,
            )
        )
        result = _get_audio_parameters(params)
        self.assertEqual(result, expected)

    def test_get_audio_parameters_long_params_shadow_short_ones(self):
        expected = (8000, 2, 1)
        params = dict(
            zip(("sampling_rate", "sample_width", "channels"), expected)
        )
        params.update(dict(zip(("sr", "sw", "ch"), "xxx")))
        result = _get_audio_parameters(params)
        self.assertEqual(result, expected)

    @genty_dataset(
        str_sampling_rate=(("x", 2, 1),),
        negative_sampling_rate=((-8000, 2, 1),),
        str_sample_width=((8000, "x", 1),),
        negative_sample_width=((8000, -2, 1),),
        str_channels=((8000, 2, "x"),),
        negative_channels=((8000, 2, -1),),
    )
    def test_get_audio_parameters_invalid(self, values):
        params = dict(
            zip(("sampling_rate", "sample_width", "channels"), values)
        )
        with self.assertRaises(AudioParameterError):
            _get_audio_parameters(params)

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

    def test_from_file_large_file_raw(self,):
        filename = "tests/data/test_16KHZ_mono_400Hz.raw"
        audio_source = from_file(
            filename,
            large_file=True,
            sampling_rate=16000,
            sample_width=2,
            channels=1,
        )
        self.assertIsInstance(audio_source, RawAudioSource)

    def test_from_file_large_file_wave(self,):
        filename = "tests/data/test_16KHZ_mono_400Hz.wav"
        audio_source = from_file(filename, large_file=True)
        self.assertIsInstance(audio_source, WaveAudioSource)

    def test_from_file_large_file_compressed(self,):
        filename = "tests/data/test_16KHZ_mono_400Hz.ogg"
        with self.assertRaises(AudioIOError):
            from_file(filename, large_file=True)

    @genty_dataset(
        missing_sampling_rate=("sr",),
        missing_sample_width=("sw",),
        missing_channels=("ch",),
    )
    def test_from_file_missing_audio_param(self, missing_param):
        with self.assertRaises(AudioParameterError):
            params = AUDIO_PARAMS_SHORT.copy()
            del params[missing_param]
            from_file("audio", audio_format="raw", **params)

    def test_from_file_no_pydub(self):
        with patch("auditok.io._WITH_PYDUB", False):
            with self.assertRaises(AudioIOError):
                from_file("audio", "mp3")

    @patch("auditok.io._WITH_PYDUB", True)
    @patch("auditok.io.BufferAudioSource")
    @genty_dataset(
        ogg_first_channel=("ogg", "from_ogg"),
        ogg_second_channel=("ogg", "from_ogg"),
        ogg_mix=("ogg", "from_ogg"),
        ogg_default=("ogg", "from_ogg"),
        mp3_left_channel=("mp3", "from_mp3"),
        mp3_right_channel=("mp3", "from_mp3"),
        flac_first_channel=("flac", "from_file"),
        flac_second_channel=("flac", "from_file"),
        flv_left_channel=("flv", "from_flv"),
        webm_right_channel=("webm", "from_file"),
    )
    def test_from_file_multichannel_audio_compressed(
        self, audio_format, function, *mocks
    ):
        filename = "audio.{}".format(audio_format)
        segment_mock = Mock()
        segment_mock.sample_width = 2
        segment_mock.channels = 2
        segment_mock._data = b"abcd"
        with patch("auditok.io.AudioSegment.{}".format(function)) as open_func:
            open_func.return_value = segment_mock
            from_file(filename)
            self.assertTrue(open_func.called)

    @genty_dataset(
        mono=("mono_400", (400,)),
        three_channel=("3channel_400-800-1600", (400, 800, 1600)),
        mono_large_file=("mono_400", (400,), True),
        three_channel_large_file=(
            "3channel_400-800-1600",
            (400, 800, 1600),
            True,
        ),
    )
    def test_load_raw(self, file_id, frequencies, large_file=False):
        filename = "tests/data/test_16KHZ_{}Hz.raw".format(file_id)
        audio_source = _load_raw(
            filename, 16000, 2, len(frequencies), large_file=large_file
        )
        audio_source.open()
        data = audio_source.read(-1)
        audio_source.close()
        expected_class = RawAudioSource if large_file else BufferAudioSource
        self.assertIsInstance(audio_source, expected_class)
        self.assertEqual(audio_source.sampling_rate, 16000)
        self.assertEqual(audio_source.sample_width, 2)
        self.assertEqual(audio_source.channels, len(frequencies))
        mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
        fmt = FORMAT[audio_source.sample_width]
        expected = array(fmt, _sample_generator(*mono_channels)).tobytes()
        self.assertEqual(data, expected)

    @genty_dataset(
        missing_sampling_rate=("sr",),
        missing_sample_width=("sw",),
        missing_channels=("ch",),
    )
    def test_load_raw_missing_audio_param(self, missing_param):
        with self.assertRaises(AudioParameterError):
            params = AUDIO_PARAMS_SHORT.copy()
            del params[missing_param]
            srate, swidth, channels, _ = _get_audio_parameters(params)
            _load_raw("audio", srate, swidth, channels)

    @genty_dataset(
        mono=("mono_400", (400,)),
        three_channel=("3channel_400-800-1600", (400, 800, 1600)),
        mono_large_file=("mono_400", (400,), True),
        three_channel_large_file=(
            "3channel_400-800-1600",
            (400, 800, 1600),
            True,
        ),
    )
    def test_load_wave(self, file_id, frequencies, large_file=False):
        filename = "tests/data/test_16KHZ_{}Hz.wav".format(file_id)
        audio_source = _load_wave(filename, large_file=large_file)
        audio_source.open()
        data = audio_source.read(-1)
        audio_source.close()
        expected_class = WaveAudioSource if large_file else BufferAudioSource
        self.assertIsInstance(audio_source, expected_class)
        self.assertEqual(audio_source.sampling_rate, 16000)
        self.assertEqual(audio_source.sample_width, 2)
        self.assertEqual(audio_source.channels, len(frequencies))
        mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
        fmt = FORMAT[audio_source.sample_width]
        expected = array(fmt, _sample_generator(*mono_channels)).tobytes()
        self.assertEqual(data, expected)

    @patch("auditok.io._WITH_PYDUB", True)
    @patch("auditok.io.BufferAudioSource")
    @genty_dataset(
        ogg_default_first_channel=("ogg", 2, "from_ogg"),
        ogg_first_channel=("ogg", 1, "from_ogg"),
        ogg_second_channel=("ogg", 2, "from_ogg"),
        ogg_mix_channels=("ogg", 3, "from_ogg"),
        mp3_left_channel=("mp3", 1, "from_mp3"),
        mp3_right_channel=("mp3", 2, "from_mp3"),
        mp3_mix_channels=("mp3", 3, "from_mp3"),
        flac_first_channel=("flac", 2, "from_file"),
        flac_second_channel=("flac", 2, "from_file"),
        flv_left_channel=("flv", 1, "from_flv"),
        webm_right_channel=("webm", 2, "from_file"),
        webm_mix_channels=("webm", 4, "from_file"),
    )
    def test_load_with_pydub(self, audio_format, channels, function, *mocks):
        filename = "audio.{}".format(audio_format)
        segment_mock = Mock()
        segment_mock.sample_width = 2
        segment_mock.channels = channels
        segment_mock._data = b"abcdefgh"
        with patch("auditok.io.AudioSegment.{}".format(function)) as open_func:
            open_func.return_value = segment_mock
            _load_with_pydub(filename, audio_format)
            self.assertTrue(open_func.called)

    @genty_dataset(
        mono=("mono_400Hz.raw", (400,)),
        three_channel=("3channel_400-800-1600Hz.raw", (400, 800, 1600)),
    )
    def test_save_raw(self, filename, frequencies):
        filename = "tests/data/test_16KHZ_{}".format(filename)
        sample_width = 2
        fmt = FORMAT[sample_width]
        mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
        data = array(fmt, _sample_generator(*mono_channels)).tobytes()
        tmpfile = NamedTemporaryFile()
        _save_raw(data, tmpfile.name)
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
        fmt = FORMAT[sample_width]
        mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
        data = array(fmt, _sample_generator(*mono_channels)).tobytes()
        tmpfile = NamedTemporaryFile()
        _save_wave(data, tmpfile.name, sampling_rate, sample_width, channels)
        self.assertTrue(filecmp.cmp(tmpfile.name, filename, shallow=False))

    @genty_dataset(
        missing_sampling_rate=("sr",),
        missing_sample_width=("sw",),
        missing_channels=("ch",),
    )
    def test_save_wave_missing_audio_param(self, missing_param):
        with self.assertRaises(AudioParameterError):
            params = AUDIO_PARAMS_SHORT.copy()
            del params[missing_param]
            srate, swidth, channels, _ = _get_audio_parameters(params)
            _save_wave(b"\0\0", "audio", srate, swidth, channels)

    def test_save_with_pydub(self):
        with patch("auditok.io.AudioSegment.export") as export:
            tmpdir = TemporaryDirectory()
            filename = os.path.join(tmpdir.name, "audio.ogg")
            _save_with_pydub(b"\0\0", filename, "ogg", 16000, 2, 1)
            self.assertTrue(export.called)
            tmpdir.cleanup()

    @genty_dataset(
        raw_with_audio_format=("audio", "raw"),
        raw_with_extension=("audio.raw", None),
        raw_with_audio_format_and_extension=("audio.mp3", "raw"),
        raw_no_audio_format_nor_extension=("audio", None),
    )
    def test_to_file_raw(self, filename, audio_format):
        exp_filename = "tests/data/test_16KHZ_mono_400Hz.raw"
        tmpdir = TemporaryDirectory()
        filename = os.path.join(tmpdir.name, filename)
        data = PURE_TONE_DICT[400].tobytes()
        to_file(data, filename, audio_format=audio_format)
        self.assertTrue(filecmp.cmp(filename, exp_filename, shallow=False))
        tmpdir.cleanup()

    @genty_dataset(
        wav_with_audio_format=("audio", "wav"),
        wav_with_extension=("audio.wav", None),
        wav_with_audio_format_and_extension=("audio.mp3", "wav"),
        wave_with_audio_format=("audio", "wave"),
        wave_with_extension=("audio.wave", None),
        wave_with_audio_format_and_extension=("audio.mp3", "wave"),
    )
    def test_to_file_wave(self, filename, audio_format):
        exp_filename = "tests/data/test_16KHZ_mono_400Hz.wav"
        tmpdir = TemporaryDirectory()
        filename = os.path.join(tmpdir.name, filename)
        data = PURE_TONE_DICT[400].tobytes()
        to_file(
            data,
            filename,
            audio_format=audio_format,
            sampling_rate=16000,
            sample_width=2,
            channels=1,
        )
        self.assertTrue(filecmp.cmp(filename, exp_filename, shallow=False))
        tmpdir.cleanup()

    @genty_dataset(
        missing_sampling_rate=("sr",),
        missing_sample_width=("sw",),
        missing_channels=("ch",),
    )
    def test_to_file_missing_audio_param(self, missing_param):
        params = AUDIO_PARAMS_SHORT.copy()
        del params[missing_param]
        with self.assertRaises(AudioParameterError):
            to_file(b"\0\0", "audio", audio_format="wav", **params)
        with self.assertRaises(AudioParameterError):
            to_file(b"\0\0", "audio", audio_format="mp3", **params)

    def test_to_file_no_pydub(self):
        with patch("auditok.io._WITH_PYDUB", False):
            with self.assertRaises(AudioIOError):
                to_file("audio", b"", "mp3")

    @patch("auditok.io._WITH_PYDUB", True)
    @genty_dataset(
        ogg_with_extension=("audio.ogg", None),
        ogg_with_audio_format=("audio", "ogg"),
        ogg_format_with_wrong_extension=("audio.wav", "ogg"),
    )
    def test_to_file_compressed(self, filename, audio_format, *mocks):
        with patch("auditok.io.AudioSegment.export") as export:
            tmpdir = TemporaryDirectory()
            filename = os.path.join(tmpdir.name, filename)
            to_file(b"\0\0", filename, audio_format, **AUDIO_PARAMS_SHORT)
            self.assertTrue(export.called)
            tmpdir.cleanup()

    @genty_dataset(
        string_wave=(
            "tests/data/test_16KHZ_mono_400Hz.wav",
            BufferAudioSource,
        ),
        string_wave_large_file=(
            "tests/data/test_16KHZ_mono_400Hz.wav",
            WaveAudioSource,
            {"large_file": True},
        ),
        stdin=("-", StdinAudioSource),
        string_raw=("tests/data/test_16KHZ_mono_400Hz.raw", BufferAudioSource),
        string_raw_large_file=(
            "tests/data/test_16KHZ_mono_400Hz.raw",
            RawAudioSource,
            {"large_file": True},
        ),
        bytes_=(b"0" * 8000, BufferAudioSource),
    )
    def test_get_audio_source(self, input, expected_type, extra_args=None):
        kwargs = {"sampling_rate": 16000, "sample_width": 2, "channels": 1}
        if extra_args is not None:
            kwargs.update(extra_args)
        audio_source = get_audio_source(input, **kwargs)
        self.assertIsInstance(audio_source, expected_type)


if __name__ == "__main__":
    unittest.main()
