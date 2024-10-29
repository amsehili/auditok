import filecmp
import os
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pytest
from test_AudioSource import PURE_TONE_DICT, _sample_generator

import auditok
from auditok.io import (
    AudioIOError,
    AudioParameterError,
    BufferAudioSource,
    RawAudioSource,
    StdinAudioSource,
    WaveAudioSource,
    _get_audio_parameters,
    _guess_audio_format,
    _load_raw,
    _load_wave,
    _load_with_pydub,
    _save_raw,
    _save_wave,
    _save_with_pydub,
    check_audio_data,
    from_file,
    get_audio_source,
    to_file,
)
from auditok.signal import SAMPLE_WIDTH_TO_DTYPE

AUDIO_PARAMS = {"sampling_rate": 16000, "sample_width": 2, "channels": 1}
AUDIO_PARAMS_SHORT = {"sr": 16000, "sw": 2, "ch": 1}


@pytest.mark.parametrize(
    "data, sample_width, channels, valid",
    [
        (b"\0" * 113, 1, 1, True),  # valid_mono
        (b"\0" * 160, 1, 2, True),  # valid_stereo
        (b"\0" * 113, 2, 1, False),  # invalid_mono_sw_2
        (b"\0" * 113, 1, 2, False),  # invalid_stereo_sw_1
        (b"\0" * 158, 2, 2, False),  # invalid_stereo_sw_2
    ],
    ids=[
        "valid_mono",
        "valid_stereo",
        "invalid_mono_sw_2",
        "invalid_stereo_sw_1",
        "invalid_stereo_sw_2",
    ],
)
def test_check_audio_data(data, sample_width, channels, valid):
    if not valid:
        with pytest.raises(AudioParameterError):
            check_audio_data(data, sample_width, channels)
    else:
        assert check_audio_data(data, sample_width, channels) is None


@pytest.mark.parametrize(
    "filename, audio_format, expected",
    [
        ("filename.wav", "wav", "wav"),  # extension_and_format_same
        ("filename.mp3", "wav", "wav"),  # extension_and_format_different
        ("filename.wav", None, "wav"),  # extension_no_format
        ("filename", "wav", "wav"),  # format_no_extension
        ("filename", None, None),  # no_format_no_extension
        ("filename", "wave", "wav"),  # wave_as_wav
        ("filename.wave", None, "wav"),  # wave_as_wav_extension
    ],
    ids=[
        "extension_and_format_same",
        "extension_and_format_different",
        "extension_no_format",
        "format_no_extension",
        "no_format_no_extension",
        "wave_as_wav",
        "wave_as_wav_extension",
    ],
)
def test_guess_audio_format(filename, audio_format, expected):
    result = _guess_audio_format(filename, audio_format)
    assert result == expected

    result = _guess_audio_format(Path(filename), audio_format)
    assert result == expected


def test_get_audio_parameters_short_params():
    expected = (8000, 2, 1)
    params = dict(zip(("sr", "sw", "ch"), expected))
    result = _get_audio_parameters(params)
    assert result == expected


def test_get_audio_parameters_long_params():
    expected = (8000, 2, 1)
    params = dict(zip(("sampling_rate", "sample_width", "channels"), expected))
    result = _get_audio_parameters(params)
    assert result == expected


def test_get_audio_parameters_long_params_shadow_short_ones():
    expected = (8000, 2, 1)
    params = dict(
        zip(
            ("sampling_rate", "sample_width", "channels"),
            expected,
        )
    )
    params.update(
        dict(
            zip(
                ("sr", "sw", "ch"),
                "xxx",
            )
        )
    )
    result = _get_audio_parameters(params)
    assert result == expected


@pytest.mark.parametrize(
    "missing_param",
    [
        "sampling_rate",  # missing_sampling_rate
        "sample_width",  # missing_sample_width
        "channels",  # missing_channels
    ],
    ids=["missing_sampling_rate", "missing_sample_width", "missing_channels"],
)
def test_get_audio_parameters_missing_parameter(missing_param):
    params = AUDIO_PARAMS.copy()
    del params[missing_param]
    with pytest.raises(AudioParameterError):
        _get_audio_parameters(params)


@pytest.mark.parametrize(
    "missing_param",
    [
        "sr",  # missing_sampling_rate
        "sw",  # missing_sample_width
        "ch",  # missing_channels
    ],
    ids=["missing_sampling_rate", "missing_sample_width", "missing_channels"],
)
def test_get_audio_parameters_missing_parameter_short(missing_param):
    params = AUDIO_PARAMS_SHORT.copy()
    del params[missing_param]
    with pytest.raises(AudioParameterError):
        _get_audio_parameters(params)


@pytest.mark.parametrize(
    "values",
    [
        ("x", 2, 1),  # str_sampling_rate
        (-8000, 2, 1),  # negative_sampling_rate
        (8000, "x", 1),  # str_sample_width
        (8000, -2, 1),  # negative_sample_width
        (8000, 2, "x"),  # str_channels
        (8000, 2, -1),  # negative_channels
    ],
    ids=[
        "str_sampling_rate",
        "negative_sampling_rate",
        "str_sample_width",
        "negative_sample_width",
        "str_channels",
        "negative_channels",
    ],
)
def test_get_audio_parameters_invalid(values):
    params = dict(
        zip(
            ("sampling_rate", "sample_width", "channels"),
            values,
        )
    )
    with pytest.raises(AudioParameterError):
        _get_audio_parameters(params)


@pytest.mark.parametrize(
    "filename, audio_format, funtion_name, kwargs",
    [
        (
            "audio",
            "raw",
            "_load_raw",
            AUDIO_PARAMS_SHORT,
        ),  # raw_with_audio_format
        (
            "audio.raw",
            None,
            "_load_raw",
            AUDIO_PARAMS_SHORT,
        ),  # raw_with_extension
        ("audio", "wave", "_load_wave", None),  # wave_with_audio_format
        ("audio", "wave", "_load_wave", None),  # wav_with_audio_format
        ("audio.wav", None, "_load_wave", None),  # wav_with_extension
        (
            "audio.dat",
            "wav",
            "_load_wave",
            None,
        ),  # format_and_extension_both_given_a
        (
            "audio.raw",
            "wave",
            "_load_wave",
            None,
        ),  # format_and_extension_both_given_b
        ("audio", None, "_load_with_pydub", None),  # no_format_nor_extension
        ("audio.ogg", None, "_load_with_pydub", None),  # other_formats_ogg
        ("audio", "webm", "_load_with_pydub", None),  # other_formats_webm
    ],
    ids=[
        "raw_with_audio_format",
        "raw_with_extension",
        "wave_with_audio_format",
        "wav_with_audio_format",
        "wav_with_extension",
        "format_and_extension_both_given_a",
        "format_and_extension_both_given_b",
        "no_format_nor_extension",
        "other_formats_ogg",
        "other_formats_webm",
    ],
)
def test_from_file(filename, audio_format, funtion_name, kwargs):
    funtion_name = "auditok.io." + funtion_name
    if kwargs is None:
        kwargs = {}
    with patch(funtion_name) as patch_function:
        from_file(filename, audio_format, **kwargs)
    assert patch_function.called


@pytest.mark.parametrize(
    "large_file, cls, size, use_pathlib",
    [
        (False, BufferAudioSource, -1, False),  # large_file_false_negative_size
        (False, BufferAudioSource, None, False),  # large_file_false_None_size
        (
            False,
            BufferAudioSource,
            None,
            True,
        ),  # large_file_false_None_size_Path
        (True, RawAudioSource, -1, False),  # large_file_true_negative_size
        (True, RawAudioSource, None, False),  # large_file_true_None_size
        (True, RawAudioSource, -1, True),  # large_file_true_negative_size_Path
    ],
    ids=[
        "large_file_false_negative_size",
        "large_file_false_None_size",
        "large_file_false_None_size_Path",
        "large_file_true_negative_size",
        "large_file_true_None_size",
        "large_file_true_negative_size_Path",
    ],
)
def test_from_file_raw_read_all(large_file, cls, size, use_pathlib):
    filename = Path("tests/data/test_16KHZ_mono_400Hz.raw")
    if use_pathlib:
        filename = Path(filename)
    audio_source = from_file(
        filename,
        large_file=large_file,
        sampling_rate=16000,
        sample_width=2,
        channels=1,
    )
    assert isinstance(audio_source, cls)

    with open(filename, "rb") as fp:
        expected = fp.read()
    audio_source.open()
    data = audio_source.read(size)
    audio_source.close()
    assert data == expected


@pytest.mark.parametrize(
    "large_file, cls, size, use_pathlib",
    [
        (False, BufferAudioSource, -1, False),  # large_file_false_negative_size
        (False, BufferAudioSource, None, False),  # large_file_false_None_size
        (
            False,
            BufferAudioSource,
            None,
            True,
        ),  # large_file_false_None_size_Path
        (True, WaveAudioSource, -1, False),  # large_file_true_negative_size
        (True, WaveAudioSource, None, False),  # large_file_true_None_size
        (True, WaveAudioSource, -1, True),  # large_file_true_negative_size_Path
    ],
    ids=[
        "large_file_false_negative_size",
        "large_file_false_None_size",
        "large_file_false_None_size_Path",
        "large_file_true_negative_size",
        "large_file_true_None_size",
        "large_file_true_negative_size_Path",
    ],
)
def test_from_file_wave_read_all(large_file, cls, size, use_pathlib):
    filename = "tests/data/test_16KHZ_mono_400Hz.wav"
    if use_pathlib:
        filename = Path(filename)
    audio_source = from_file(
        filename,
        large_file=large_file,
        sampling_rate=16000,
        sample_width=2,
        channels=1,
    )
    assert isinstance(audio_source, cls)

    with wave.open(str(filename)) as fp:
        expected = fp.readframes(-1)
    audio_source.open()
    data = audio_source.read(size)
    audio_source.close()
    assert data == expected


def test_from_file_large_file_compressed():
    filename = "tests/data/test_16KHZ_mono_400Hz.ogg"
    with pytest.raises(AudioIOError):
        from_file(filename, large_file=True)


@pytest.mark.parametrize(
    "missing_param",
    [
        "sr",  # missing_sampling_rate
        "sw",  # missing_sample_width
        "ch",  # missing_channels
    ],
    ids=["missing_sampling_rate", "missing_sample_width", "missing_channels"],
)
def test_from_file_missing_audio_param(missing_param):
    params = AUDIO_PARAMS_SHORT.copy()
    del params[missing_param]
    with pytest.raises(AudioParameterError):
        from_file("audio", audio_format="raw", **params)


def test_from_file_no_pydub():
    with patch("auditok.io._WITH_PYDUB", False):
        with pytest.raises(AudioIOError):
            from_file("audio", "mp3")


@pytest.mark.parametrize(
    "audio_format, function",
    [
        ("ogg", "from_ogg"),  # ogg_first_channel
        ("ogg", "from_ogg"),  # ogg_second_channel
        ("ogg", "from_ogg"),  # ogg_mix
        ("ogg", "from_ogg"),  # ogg_default
        ("mp3", "from_mp3"),  # mp3_left_channel
        ("mp3", "from_mp3"),  # mp3_right_channel
        ("flac", "from_file"),  # flac_first_channel
        ("flac", "from_file"),  # flac_second_channel
        ("flv", "from_flv"),  # flv_left_channel
        ("webm", "from_file"),  # webm_right_channel
    ],
    ids=[
        "ogg_first_channel",
        "ogg_second_channel",
        "ogg_mix",
        "ogg_default",
        "mp3_left_channel",
        "mp3_right_channel",
        "flac_first_channel",
        "flac_second_channel",
        "flv_left_channel",
        "webm_right_channel",
    ],
)
@patch("auditok.io._WITH_PYDUB", True)
@patch("auditok.io.BufferAudioSource")
def test_from_file_multichannel_audio_compressed(
    mock_buffer_audio_source, audio_format, function
):
    filename = "audio.{}".format(audio_format)
    segment_mock = Mock()
    segment_mock.sample_width = 2
    segment_mock.channels = 2
    segment_mock._data = b"abcd"
    with patch("auditok.io.AudioSegment.{}".format(function)) as open_func:
        open_func.return_value = segment_mock
        from_file(filename)
        assert open_func.called


@pytest.mark.parametrize(
    "file_id, frequencies, large_file",
    [
        ("mono_400", (400,), False),  # mono
        ("3channel_400-800-1600", (400, 800, 1600), False),  # three_channel
        ("mono_400", (400,), True),  # mono_large_file
        (
            "3channel_400-800-1600",
            (400, 800, 1600),
            True,
        ),  # three_channel_large_file
    ],
    ids=[
        "mono",
        "three_channel",
        "mono_large_file",
        "three_channel_large_file",
    ],
)
def test_load_raw(file_id, frequencies, large_file):
    filename = "tests/data/test_16KHZ_{}Hz.raw".format(file_id)
    audio_source = _load_raw(
        filename, 16000, 2, len(frequencies), large_file=large_file
    )
    audio_source.open()
    data = audio_source.read(-1)
    audio_source.close()
    expected_class = RawAudioSource if large_file else BufferAudioSource
    assert isinstance(audio_source, expected_class)
    assert audio_source.sampling_rate == 16000
    assert audio_source.sample_width == 2
    assert audio_source.channels == len(frequencies)
    mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
    dtype = SAMPLE_WIDTH_TO_DTYPE[audio_source.sample_width]
    expected = np.fromiter(
        _sample_generator(*mono_channels), dtype=dtype
    ).tobytes()
    assert data == expected


def test_load_raw_missing_audio_param():
    with pytest.raises(AudioParameterError):
        _load_raw("audio", sampling_rate=None, sample_width=1, channels=1)

    with pytest.raises(AudioParameterError):
        _load_raw("audio", sampling_rate=16000, sample_width=None, channels=1)

    with pytest.raises(AudioParameterError):
        _load_raw("audio", sampling_rate=16000, sample_width=1, channels=None)


@pytest.mark.parametrize(
    "file_id, frequencies, large_file",
    [
        ("mono_400", (400,), False),  # mono
        ("3channel_400-800-1600", (400, 800, 1600), False),  # three_channel
        ("mono_400", (400,), True),  # mono_large_file
        (
            "3channel_400-800-1600",
            (400, 800, 1600),
            True,
        ),  # three_channel_large_file
    ],
    ids=[
        "mono",
        "three_channel",
        "mono_large_file",
        "three_channel_large_file",
    ],
)
def test_load_wave(file_id, frequencies, large_file):
    filename = "tests/data/test_16KHZ_{}Hz.wav".format(file_id)
    audio_source = _load_wave(filename, large_file=large_file)
    audio_source.open()
    data = audio_source.read(-1)
    audio_source.close()
    expected_class = WaveAudioSource if large_file else BufferAudioSource
    assert isinstance(audio_source, expected_class)
    assert audio_source.sampling_rate == 16000
    assert audio_source.sample_width == 2
    assert audio_source.channels == len(frequencies)
    mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
    dtype = SAMPLE_WIDTH_TO_DTYPE[audio_source.sample_width]
    expected = np.fromiter(
        _sample_generator(*mono_channels), dtype=dtype
    ).tobytes()
    assert data == expected


@pytest.mark.parametrize(
    "audio_format, channels, function",
    [
        ("ogg", 2, "from_ogg"),  # ogg_default_first_channel
        ("ogg", 1, "from_ogg"),  # ogg_first_channel
        ("ogg", 2, "from_ogg"),  # ogg_second_channel
        ("ogg", 3, "from_ogg"),  # ogg_mix_channels
        ("mp3", 1, "from_mp3"),  # mp3_left_channel
        ("mp3", 2, "from_mp3"),  # mp3_right_channel
        ("mp3", 3, "from_mp3"),  # mp3_mix_channels
        ("flac", 2, "from_file"),  # flac_first_channel
        ("flac", 2, "from_file"),  # flac_second_channel
        ("flv", 1, "from_flv"),  # flv_left_channel
        ("webm", 2, "from_file"),  # webm_right_channel
        ("webm", 4, "from_file"),  # webm_mix_channels
    ],
    ids=[
        "ogg_default_first_channel",
        "ogg_first_channel",
        "ogg_second_channel",
        "ogg_mix_channels",
        "mp3_left_channel",
        "mp3_right_channel",
        "mp3_mix_channels",
        "flac_first_channel",
        "flac_second_channel",
        "flv_left_channel",
        "webm_right_channel",
        "webm_mix_channels",
    ],
)
@patch("auditok.io._WITH_PYDUB", True)
@patch("auditok.io.BufferAudioSource")
def test_load_with_pydub(
    mock_buffer_audio_source, audio_format, channels, function
):
    filename = "audio.{}".format(audio_format)
    segment_mock = Mock()
    segment_mock.sample_width = 2
    segment_mock.channels = channels
    segment_mock._data = b"abcdefgh"
    with patch("auditok.io.AudioSegment.{}".format(function)) as open_func:
        open_func.return_value = segment_mock
        _load_with_pydub(filename, audio_format)
        assert open_func.called


@pytest.mark.parametrize(
    "filename, frequencies, use_pathlib",
    [
        ("mono_400Hz.raw", (400,), False),  # mono
        ("mono_400Hz.raw", (400,), True),  # mono_pathlib
        (
            "3channel_400-800-1600Hz.raw",
            (400, 800, 1600),
            False,
        ),  # three_channel
    ],
    ids=["mono", "three_channel", "use_pathlib"],
)
def test_save_raw(filename, frequencies, use_pathlib):
    filename = "tests/data/test_16KHZ_{}".format(filename)
    if use_pathlib:
        filename = Path(filename)
    sample_width = 2
    dtype = SAMPLE_WIDTH_TO_DTYPE[sample_width]
    mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
    data = np.fromiter(_sample_generator(*mono_channels), dtype=dtype).tobytes()
    tmpfile = NamedTemporaryFile()
    _save_raw(data, tmpfile.name)
    assert filecmp.cmp(tmpfile.name, filename, shallow=False)


@pytest.mark.parametrize(
    "filename, frequencies, use_pathlib",
    [
        ("mono_400Hz.wav", (400,), False),  # mono
        ("mono_400Hz.wav", (400,), True),  # mono_pathlib
        (
            "3channel_400-800-1600Hz.wav",
            (400, 800, 1600),
            False,
        ),  # three_channel
    ],
    ids=["mono", "mono_pathlib", "three_channel"],
)
def test_save_wave(filename, frequencies, use_pathlib):
    filename = "tests/data/test_16KHZ_{}".format(filename)
    if use_pathlib:
        filename = str(filename)
    sampling_rate = 16000
    sample_width = 2
    channels = len(frequencies)
    mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
    dtype = SAMPLE_WIDTH_TO_DTYPE[sample_width]
    data = np.fromiter(_sample_generator(*mono_channels), dtype=dtype).tobytes()
    tmpfile = NamedTemporaryFile()
    _save_wave(data, tmpfile.name, sampling_rate, sample_width, channels)
    assert filecmp.cmp(tmpfile.name, filename, shallow=False)


@pytest.mark.parametrize(
    "missing_param",
    [
        "sr",  # missing_sampling_rate
        "sw",  # missing_sample_width
        "ch",  # missing_channels
    ],
    ids=["missing_sampling_rate", "missing_sample_width", "missing_channels"],
)
def test_save_wave_missing_audio_param(missing_param):
    with pytest.raises(AudioParameterError):
        _save_wave(
            b"\0\0", "audio", sampling_rate=None, sample_width=1, channels=1
        )

    with pytest.raises(AudioParameterError):
        _save_wave(
            b"\0\0", "audio", sampling_rate=16000, sample_width=None, channels=1
        )

    with pytest.raises(AudioParameterError):
        _save_wave(
            b"\0\0", "audio", sampling_rate=16000, sample_width=1, channels=None
        )


def test_save_with_pydub():
    with patch("auditok.io.AudioSegment.export") as export:
        tmpdir = TemporaryDirectory()
        filename = os.path.join(tmpdir.name, "audio.ogg")
        _save_with_pydub(b"\0\0", filename, "ogg", 16000, 2, 1)
        assert export.called
        tmpdir.cleanup()


@pytest.mark.parametrize(
    "filename, audio_format",
    [
        ("audio", "raw"),  # raw_with_audio_format
        ("audio.raw", None),  # raw_with_extension
        ("audio.mp3", "raw"),  # raw_with_audio_format_and_extension
        ("audio", None),  # raw_no_audio_format_nor_extension
    ],
    ids=[
        "raw_with_audio_format",
        "raw_with_extension",
        "raw_with_audio_format_and_extension",
        "raw_no_audio_format_nor_extension",
    ],
)
def test_to_file_raw(filename, audio_format):
    exp_filename = "tests/data/test_16KHZ_mono_400Hz.raw"
    tmpdir = TemporaryDirectory()
    filename = os.path.join(tmpdir.name, filename)
    data = PURE_TONE_DICT[400].tobytes()
    to_file(data, filename, audio_format=audio_format)
    assert filecmp.cmp(filename, exp_filename, shallow=False)
    tmpdir.cleanup()


@pytest.mark.parametrize(
    "filename, audio_format",
    [
        ("audio", "wav"),  # wav_with_audio_format
        ("audio.wav", None),  # wav_with_extension
        ("audio.mp3", "wav"),  # wav_with_audio_format_and_extension
        ("audio", "wave"),  # wave_with_audio_format
        ("audio.wave", None),  # wave_with_extension
        ("audio.mp3", "wave"),  # wave_with_audio_format_and_extension
    ],
    ids=[
        "wav_with_audio_format",
        "wav_with_extension",
        "wav_with_audio_format_and_extension",
        "wave_with_audio_format",
        "wave_with_extension",
        "wave_with_audio_format_and_extension",
    ],
)
def test_to_file_wave(filename, audio_format):
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
    assert filecmp.cmp(filename, exp_filename, shallow=False)
    tmpdir.cleanup()


@pytest.mark.parametrize(
    "missing_param",
    [
        "sr",  # missing_sampling_rate
        "sw",  # missing_sample_width
        "ch",  # missing_channels
    ],
    ids=["missing_sampling_rate", "missing_sample_width", "missing_channels"],
)
def test_to_file_missing_audio_param(missing_param):
    params = AUDIO_PARAMS_SHORT.copy()
    del params[missing_param]
    with pytest.raises(AudioParameterError):
        to_file(b"\0\0", "audio", audio_format="wav", **params)
    with pytest.raises(AudioParameterError):
        to_file(b"\0\0", "audio", audio_format="mp3", **params)


def test_to_file_no_pydub():
    with patch("auditok.io._WITH_PYDUB", False):
        with pytest.raises(AudioIOError):
            to_file("audio", b"", "mp3")


@pytest.mark.parametrize(
    "filename, audio_format",
    [
        ("audio.ogg", None),  # ogg_with_extension
        ("audio", "ogg"),  # ogg_with_audio_format
        ("audio.wav", "ogg"),  # ogg_format_with_wrong_extension
    ],
    ids=[
        "ogg_with_extension",
        "ogg_with_audio_format",
        "ogg_format_with_wrong_extension",
    ],
)
@patch("auditok.io._WITH_PYDUB", True)
def test_to_file_compressed(filename, audio_format):
    with patch("auditok.io.AudioSegment.export") as export:
        tmpdir = TemporaryDirectory()
        filename = os.path.join(tmpdir.name, filename)
        to_file(b"\0\0", filename, audio_format, **AUDIO_PARAMS_SHORT)
        assert export.called
        tmpdir.cleanup()


@pytest.mark.parametrize(
    "input, expected_type, extra_args",
    [
        (
            "tests/data/test_16KHZ_mono_400Hz.wav",
            BufferAudioSource,
            None,
        ),  # string_wave
        (
            "tests/data/test_16KHZ_mono_400Hz.wav",
            WaveAudioSource,
            {"large_file": True},
        ),  # string_wave_large_file
        ("-", StdinAudioSource, None),  # stdin
        (
            "tests/data/test_16KHZ_mono_400Hz.raw",
            BufferAudioSource,
            None,
        ),  # string_raw
        (
            "tests/data/test_16KHZ_mono_400Hz.raw",
            RawAudioSource,
            {"large_file": True},
        ),  # string_raw_large_file
        (b"0" * 8000, BufferAudioSource, None),  # bytes_
    ],
    ids=[
        "string_wave",
        "string_wave_large_file",
        "stdin",
        "string_raw",
        "string_raw_large_file",
        "bytes_",
    ],
)
def test_get_audio_source(input, expected_type, extra_args):
    kwargs = {"sampling_rate": 16000, "sample_width": 2, "channels": 1}
    if extra_args is not None:
        kwargs.update(extra_args)
    audio_source = get_audio_source(input, **kwargs)
    assert isinstance(audio_source, expected_type)
    assert audio_source.sampling_rate == 16000, (
        "Unexpected sampling rate: audio_source.sampling_rate = "
        + f"{audio_source.sampling_rate} instead of 16000"
    )
    assert audio_source.sr == 16000, (
        "Unexpected sampling rate: audio_source.sr = "
        + f"{audio_source.sr} instead of 16000"
    )
    assert audio_source.sample_width == 2, (
        "Unexpected sample width: audio_source.sample_width = "
        + f"{audio_source.sample_width} instead of 2"
    )
    assert audio_source.sw == 2, (
        "Unexpected sample width: audio_source.sw = "
        + f"{audio_source.sw} instead of 2"
    )
    assert audio_source.channels == 1, (
        "Unexpected number of channels: audio_source.channels = "
        + f"{audio_source.channels} instead of 1"
    )
    assert audio_source.ch == 1, (
        "Unexpected number of channels: audio_source.ch = "
        + f"{audio_source.ch} instead of 1"
    )


def test_get_audio_source_alias_prams():
    audio_source = get_audio_source(b"0" * 1600, sr=16000, sw=2, ch=1)
    assert audio_source.sampling_rate == 16000, (
        "Unexpected sampling rate: audio_source.sampling_rate = "
        + f"{audio_source.sampling_rate} instead of 16000"
    )
    assert audio_source.sr == 16000, (
        "Unexpected sampling rate: audio_source.sr = "
        + f"{audio_source.sr} instead of 16000"
    )
    assert audio_source.sample_width == 2, (
        "Unexpected sample width: audio_source.sample_width = "
        + f"{audio_source.sample_width} instead of 2"
    )
    assert audio_source.sw == 2, (
        "Unexpected sample width: audio_source.sw = "
        + f"{audio_source.sw} instead of 2"
    )
    assert audio_source.channels == 1, (
        "Unexpected number of channels: audio_source.channels = "
        + f"{audio_source.channels} instead of 1"
    )
    assert audio_source.ch == 1, (
        "Unexpected number of channels: audio_source.ch = "
        + f"{audio_source.ch} instead of 1"
    )
