import math
import os
from pathlib import Path
from random import random
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pytest

from auditok import (
    AudioParameterError,
    AudioRegion,
    load,
    make_silence,
    split,
    split_and_join_with_silence,
)
from auditok.core import (
    _duration_to_nb_windows,
    _make_audio_region,
    _read_chunks_online,
    _read_offline,
)
from auditok.io import get_audio_source
from auditok.signal import to_array
from auditok.util import AudioReader


def _make_random_length_regions(
    byte_seq, sampling_rate, sample_width, channels
):
    regions = []
    for b in byte_seq:
        duration = round(random() * 10, 6)
        data = b * int(duration * sampling_rate) * sample_width * channels
        region = AudioRegion(data, sampling_rate, sample_width, channels)
        regions.append(region)
    return regions


@pytest.mark.parametrize(
    "skip, max_read, channels",
    [
        (0, -1, 1),  # no_skip_read_all
        (0, -1, 2),  # no_skip_read_all_stereo
        (2, -1, 1),  # skip_2_read_all
        (2, None, 1),  # skip_2_read_all_None
        (2, 3, 1),  # skip_2_read_3
        (2, 3.5, 2),  # skip_2_read_3_5_stereo
        (2.4, 3.5, 2),  # skip_2_4_read_3_5_stereo
    ],
    ids=[
        "no_skip_read_all",
        "no_skip_read_all_stereo",
        "skip_2_read_all",
        "skip_2_read_all_None",
        "skip_2_read_3",
        "skip_2_read_3_5_stereo",
        "skip_2_4_read_3_5_stereo",
    ],
)
def test_load(skip, max_read, channels):
    sampling_rate = 10
    sample_width = 2
    filename = "tests/data/test_split_10HZ_{}.raw"
    filename = filename.format("mono" if channels == 1 else "stereo")
    region = load(
        filename,
        skip=skip,
        max_read=max_read,
        sr=sampling_rate,
        sw=sample_width,
        ch=channels,
    )
    with open(filename, "rb") as fp:
        fp.read(round(skip * sampling_rate * sample_width * channels))
        if max_read is None or max_read < 0:
            to_read = -1
        else:
            to_read = round(max_read * sampling_rate * sample_width * channels)
        expected = fp.read(to_read)
    assert bytes(region) == expected


@pytest.mark.parametrize(
    "duration, sampling_rate, sample_width, channels",
    [
        (1.05, 16000, 1, 1),  # mono_16K_1byte
        (1.5, 16000, 2, 1),  # mono_16K_2byte
        (1.0001, 44100, 2, 2),  # stereo_44100_2byte
        (1.000005, 48000, 2, 3),  # 3channel_48K_2byte
        (1.0001, 48000, 4, 4),  # 4channel_48K_4byte
        (0, 48000, 4, 4),  # 4channel_4K_4byte_0sec
    ],
    ids=[
        "mono_16K_1byte",
        "mono_16K_2byte",
        "stereo_44100_2byte",
        "3channel_48000_2byte",
        "4channel_48K_4byte",
        "4channel_4K_4byte_0sec",
    ],
)
def test_make_silence(duration, sampling_rate, sample_width, channels):
    silence = make_silence(duration, sampling_rate, sample_width, channels)
    size = round(duration * sampling_rate) * sample_width * channels
    expected_data = b"\0" * size
    expected_duration = size / (sampling_rate * sample_width * channels)
    assert silence.duration == expected_duration
    assert silence.data == expected_data


@pytest.mark.parametrize(
    "duration",
    [
        (0,),  # zero_second
        (1,),  # one_second
        (1.0001,),  # 1.0001_second
    ],
    ids=[
        "zero_second",
        "one_second",
        "1.0001_second",
    ],
)
def test_split_and_join_with_silence(duration):
    duration = 1.0
    sampling_rate = 10
    sample_width = 2
    channels = 1

    regions = split(
        input="tests/data/test_split_10HZ_mono.raw",
        min_dur=0.2,
        max_dur=5,
        max_silence=0.2,
        drop_trailing_silence=False,
        strict_min_dur=False,
        analysis_window=0.1,
        sr=sampling_rate,
        sw=sample_width,
        ch=channels,
        eth=50,
    )

    size = round(duration * sampling_rate) * sample_width * channels
    join_data = b"\0" * size
    expected_data = join_data.join(region.data for region in regions)
    expected_region = AudioRegion(
        expected_data, sampling_rate, sample_width, channels
    )

    region_with_silence = split_and_join_with_silence(
        input="tests/data/test_split_10HZ_mono.raw",
        silence_duration=duration,
        min_dur=0.2,
        max_dur=5,
        max_silence=0.2,
        drop_trailing_silence=False,
        strict_min_dur=False,
        analysis_window=0.1,
        sr=sampling_rate,
        sw=sample_width,
        ch=channels,
        eth=50,
    )
    assert region_with_silence == expected_region


@pytest.mark.parametrize(
    "duration, analysis_window, round_fn, expected, kwargs",
    [
        (0, 1, None, 0, None),  # zero_duration
        (0.3, 0.1, round, 3, None),  # multiple
        (0.35, 0.1, math.ceil, 4, None),  # not_multiple_ceil
        (0.35, 0.1, math.floor, 3, None),  # not_multiple_floor
        (0.05, 0.1, round, 0, None),  # small_duration
        (0.05, 0.1, math.ceil, 1, None),  # small_duration_ceil
        (0.3, 0.1, math.floor, 3, {"epsilon": 1e-6}),  # with_round_error
        (-0.5, 0.1, math.ceil, ValueError, None),  # negative_duration
        (0.5, -0.1, math.ceil, ValueError, None),  # negative_analysis_window
    ],
    ids=[
        "zero_duration",
        "multiple",
        "not_multiple_ceil",
        "not_multiple_floor",
        "small_duration",
        "small_duration_ceil",
        "with_round_error",
        "negative_duration",
        "negative_analysis_window",
    ],
)
def test_duration_to_nb_windows(
    duration, analysis_window, round_fn, expected, kwargs
):
    if expected == ValueError:
        with pytest.raises(ValueError):
            _duration_to_nb_windows(duration, analysis_window, round_fn)
    else:
        if kwargs is None:
            kwargs = {}
        result = _duration_to_nb_windows(
            duration, analysis_window, round_fn, **kwargs
        )
        assert result == expected


@pytest.mark.parametrize(
    "channels, skip, max_read",
    [
        (1, 0, None),  # mono_skip_0_max_read_None
        (1, 3, None),  # mono_skip_3_max_read_None
        (1, 2, -1),  # mono_skip_2_max_read_negative
        (1, 2, 3),  # mono_skip_2_max_read_3
        (2, 0, None),  # stereo_skip_0_max_read_None
        (2, 3, None),  # stereo_skip_3_max_read_None
        (2, 2, -1),  # stereo_skip_2_max_read_negative
        (2, 2, 3),  # stereo_skip_2_max_read_3
    ],
    ids=[
        "mono_skip_0_max_read_None",
        "mono_skip_3_max_read_None",
        "mono_skip_2_max_read_negative",
        "mono_skip_2_max_read_3",
        "stereo_skip_0_max_read_None",
        "stereo_skip_3_max_read_None",
        "stereo_skip_2_max_read_negative",
        "stereo_skip_2_max_read_3",
    ],
)
def test_read_offline(channels, skip, max_read):
    sampling_rate = 10
    sample_width = 2
    mono_or_stereo = "mono" if channels == 1 else "stereo"
    filename = "tests/data/test_split_10HZ_{}.raw".format(mono_or_stereo)
    with open(filename, "rb") as fp:
        data = fp.read()
    onset = round(skip * sampling_rate * sample_width * channels)
    if max_read in (-1, None):
        offset = len(data) + 1
    else:
        offset = onset + round(
            max_read * sampling_rate * sample_width * channels
        )
    expected_data = data[onset:offset]
    read_data, *audio_params = _read_offline(
        filename,
        skip=skip,
        max_read=max_read,
        sr=sampling_rate,
        sw=sample_width,
        ch=channels,
    )
    assert read_data == expected_data
    assert tuple(audio_params) == (sampling_rate, sample_width, channels)


@pytest.mark.parametrize(
    (
        "min_dur, max_dur, max_silence, drop_trailing_silence, "
        + "strict_min_dur, kwargs, expected"
    ),
    [
        (
            0.2,
            5,
            0.2,
            False,
            False,
            {"eth": 50},
            [(2, 16), (17, 31), (34, 76)],
        ),  # simple
        (
            0.3,
            2,
            0.2,
            False,
            False,
            {"eth": 50},
            [(2, 16), (17, 31), (34, 54), (54, 74), (74, 76)],
        ),  # short_max_dur
        (3, 5, 0.2, False, False, {"eth": 50}, [(34, 76)]),  # long_min_dur
        (0.2, 80, 10, False, False, {"eth": 50}, [(2, 76)]),  # long_max_silence
        (
            0.2,
            5,
            0.0,
            False,
            False,
            {"eth": 50},
            [(2, 14), (17, 24), (26, 29), (34, 76)],
        ),  # zero_max_silence
        (
            0.2,
            5,
            0.2,
            False,
            False,
            {"energy_threshold": 40},
            [(0, 50), (50, 76)],
        ),  # low_energy_threshold
        (
            0.2,
            5,
            0.2,
            False,
            False,
            {"energy_threshold": 60},
            [],
        ),  # high_energy_threshold
        (
            0.2,
            10,
            0.5,
            True,
            False,
            {"eth": 50},
            [(2, 76)],
        ),  # trim_leading_and_trailing_silence
        (
            0.2,
            5,
            0.2,
            True,
            False,
            {"eth": 50},
            [(2, 14), (17, 29), (34, 76)],
        ),  # drop_trailing_silence
        (
            1.5,
            5,
            0.2,
            True,
            False,
            {"eth": 50},
            [(34, 76)],
        ),  # drop_trailing_silence_2
        (
            0.3,
            2,
            0.2,
            False,
            True,
            {"eth": 50},
            [(2, 16), (17, 31), (34, 54), (54, 74)],
        ),  # strict_min_dur
    ],
    ids=[
        "simple",
        "short_max_dur",
        "long_min_dur",
        "long_max_silence",
        "zero_max_silence",
        "low_energy_threshold",
        "high_energy_threshold",
        "trim_leading_and_trailing_silence",
        "drop_trailing_silence",
        "drop_trailing_silence_2",
        "strict_min_dur",
    ],
)
def test_split_params(
    min_dur,
    max_dur,
    max_silence,
    drop_trailing_silence,
    strict_min_dur,
    kwargs,
    expected,
):
    with open("tests/data/test_split_10HZ_mono.raw", "rb") as fp:
        data = fp.read()

    regions = split(
        data,
        min_dur,
        max_dur,
        max_silence,
        drop_trailing_silence,
        strict_min_dur,
        analysis_window=0.1,
        sr=10,
        sw=2,
        ch=1,
        **kwargs
    )

    region = AudioRegion(data, 10, 2, 1)
    regions_ar = region.split(
        min_dur,
        max_dur,
        max_silence,
        drop_trailing_silence,
        strict_min_dur,
        analysis_window=0.1,
        **kwargs
    )

    regions = list(regions)
    regions_ar = list(regions_ar)
    err_msg = "Wrong number of regions after split, expected: "
    err_msg += "{}, found: {}".format(len(expected), len(regions))
    assert len(regions) == len(expected), err_msg
    err_msg = "Wrong number of regions after AudioRegion.split, expected: "
    err_msg += "{}, found: {}".format(len(expected), len(regions_ar))
    assert len(regions_ar) == len(expected), err_msg

    sample_width = 2
    for reg, reg_ar, exp in zip(regions, regions_ar, expected, strict=True):
        onset, offset = exp
        exp_data = data[onset * sample_width : offset * sample_width]
        assert bytes(reg) == exp_data
        assert reg == reg_ar


@pytest.mark.parametrize(
    "channels, kwargs, expected",
    [
        (2, {}, [(2, 32), (34, 76)]),  # stereo_all_default
        (1, {"max_read": 5}, [(2, 16), (17, 31), (34, 50)]),  # mono_max_read
        (
            1,
            {"mr": 5},
            [(2, 16), (17, 31), (34, 50)],
        ),  # mono_max_read_short_name
        (
            1,
            {"eth": 50, "use_channel": 0},
            [(2, 16), (17, 31), (34, 76)],
        ),  # mono_use_channel_1
        (1, {"eth": 50, "uc": 1}, [(2, 16), (17, 31), (34, 76)]),  # mono_uc_1
        (
            1,
            {"eth": 50, "use_channel": None},
            [(2, 16), (17, 31), (34, 76)],
        ),  # mono_use_channel_None
        (
            2,
            {"eth": 50, "use_channel": 0},
            [(2, 16), (17, 31), (34, 76)],
        ),  # stereo_use_channel_1
        (
            2,
            {"eth": 50},
            [(2, 32), (34, 76)],
        ),  # stereo_use_channel_no_use_channel_given
        (
            2,
            {"eth": 50, "use_channel": -2},
            [(2, 16), (17, 31), (34, 76)],
        ),  # stereo_use_channel_minus_2
        (2, {"eth": 50, "uc": 1}, [(10, 32), (36, 76)]),  # stereo_uc_2
        (2, {"eth": 50, "uc": -1}, [(10, 32), (36, 76)]),  # stereo_uc_minus_1
        (
            1,
            {"eth": 50, "uc": "mix"},
            [(2, 16), (17, 31), (34, 76)],
        ),  # mono_uc_mix
        (
            2,
            {"energy_threshold": 53.5, "use_channel": "mix"},
            [(54, 76)],
        ),  # stereo_use_channel_mix
        (2, {"eth": 52, "uc": "mix"}, [(17, 26), (54, 76)]),  # stereo_uc_mix
        (
            2,
            {"uc": "mix"},
            [(10, 16), (17, 31), (36, 76)],
        ),  # stereo_uc_mix_default_eth
    ],
    ids=[
        "stereo_all_default",
        "mono_max_read",
        "mono_max_read_short_name",
        "mono_use_channel_1",
        "mono_uc_1",
        "mono_use_channel_None",
        "stereo_use_channel_1",
        "stereo_use_channel_no_use_channel_given",
        "stereo_use_channel_minus_2",
        "stereo_uc_2",
        "stereo_uc_minus_1",
        "mono_uc_mix",
        "stereo_use_channel_mix",
        "stereo_uc_mix",
        "stereo_uc_mix_default_eth",
    ],
)
def test_split_kwargs(channels, kwargs, expected):

    mono_or_stereo = "mono" if channels == 1 else "stereo"
    filename = "tests/data/test_split_10HZ_{}.raw".format(mono_or_stereo)
    with open(filename, "rb") as fp:
        data = fp.read()

    regions = split(
        data,
        min_dur=0.2,
        max_dur=5,
        max_silence=0.2,
        drop_trailing_silence=False,
        strict_min_dur=False,
        analysis_window=0.1,
        sr=10,
        sw=2,
        ch=channels,
        **kwargs
    )

    region = AudioRegion(data, 10, 2, channels)
    max_read = kwargs.get("max_read", kwargs.get("mr"))
    if max_read is not None:
        region = region.sec[:max_read]
        kwargs.pop("max_read", None)
        kwargs.pop("mr", None)

    regions_ar = region.split(
        min_dur=0.2,
        max_dur=5,
        max_silence=0.2,
        drop_trailing_silence=False,
        strict_min_dur=False,
        analysis_window=0.1,
        **kwargs
    )

    regions = list(regions)
    regions_ar = list(regions_ar)
    err_msg = "Wrong number of regions after split, expected: "
    err_msg += "{}, found: {}".format(len(expected), len(regions))
    assert len(regions) == len(expected), err_msg
    err_msg = "Wrong number of regions after AudioRegion.split, expected: "
    err_msg += "{}, found: {}".format(len(expected), len(regions_ar))
    assert len(regions_ar) == len(expected), err_msg

    sample_width = 2
    sample_size_bytes = sample_width * channels
    for reg, reg_ar, exp in zip(regions, regions_ar, expected, strict=True):
        onset, offset = exp
        exp_data = data[onset * sample_size_bytes : offset * sample_size_bytes]
        assert len(bytes(reg)) == len(exp_data)
        assert reg == reg_ar


@pytest.mark.parametrize(
    "min_dur, max_dur, max_silence, channels, kwargs, expected",
    [
        (
            0.2,
            5,
            0.2,
            1,
            {"aw": 0.2},
            [(2, 30), (34, 76)],
        ),  # mono_aw_0_2_max_silence_0_2
        (
            0.2,
            5,
            0.3,
            1,
            {"aw": 0.2},
            [(2, 30), (34, 76)],
        ),  # mono_aw_0_2_max_silence_0_3
        (
            0.2,
            5,
            0.4,
            1,
            {"aw": 0.2},
            [(2, 32), (34, 76)],
        ),  # mono_aw_0_2_max_silence_0_4
        (
            0.2,
            5,
            0,
            1,
            {"aw": 0.2},
            [(2, 14), (16, 24), (26, 28), (34, 76)],
        ),  # mono_aw_0_2_max_silence_0
        (0.2, 5, 0.2, 1, {"aw": 0.2}, [(2, 30), (34, 76)]),  # mono_aw_0_2
        (
            0.3,
            5,
            0,
            1,
            {"aw": 0.3},
            [(3, 12), (15, 24), (36, 76)],
        ),  # mono_aw_0_3_max_silence_0
        (
            0.3,
            5,
            0.3,
            1,
            {"aw": 0.3},
            [(3, 27), (36, 76)],
        ),  # mono_aw_0_3_max_silence_0_3
        (
            0.3,
            5,
            0.5,
            1,
            {"aw": 0.3},
            [(3, 27), (36, 76)],
        ),  # mono_aw_0_3_max_silence_0_5
        (
            0.3,
            5,
            0.6,
            1,
            {"aw": 0.3},
            [(3, 30), (36, 76)],
        ),  # mono_aw_0_3_max_silence_0_6
        (
            0.2,
            5,
            0,
            1,
            {"aw": 0.4},
            [(4, 12), (16, 24), (36, 76)],
        ),  # mono_aw_0_4_max_silence_0
        (
            0.2,
            5,
            0.3,
            1,
            {"aw": 0.4},
            [(4, 12), (16, 24), (36, 76)],
        ),  # mono_aw_0_4_max_silence_0_3
        (
            0.2,
            5,
            0.4,
            1,
            {"aw": 0.4},
            [(4, 28), (36, 76)],
        ),  # mono_aw_0_4_max_silence_0_4
        (
            0.2,
            5,
            0.2,
            2,
            {"analysis_window": 0.2},
            [(2, 32), (34, 76)],
        ),  # stereo_uc_None_analysis_window_0_2
        (
            0.2,
            5,
            0.2,
            2,
            {"uc": None, "analysis_window": 0.2},
            [(2, 32), (34, 76)],
        ),  # stereo_uc_any_analysis_window_0_2
        (
            0.2,
            5,
            0.2,
            2,
            {"use_channel": None, "analysis_window": 0.3},
            [(3, 30), (36, 76)],
        ),  # stereo_use_channel_None_aw_0_3_max_silence_0_2
        (
            0.2,
            5,
            0.3,
            2,
            {"use_channel": "any", "analysis_window": 0.3},
            [(3, 33), (36, 76)],
        ),  # stereo_use_channel_any_aw_0_3_max_silence_0_3
        (
            0.2,
            5,
            0.2,
            2,
            {"use_channel": None, "analysis_window": 0.4},
            [(4, 28), (36, 76)],
        ),  # stereo_use_channel_None_aw_0_4_max_silence_0_2
        (
            0.2,
            5,
            0.4,
            2,
            {"use_channel": "any", "analysis_window": 0.4},
            [(4, 32), (36, 76)],
        ),  # stereo_use_channel_any_aw_0_3_max_silence_0_4
        (
            0.2,
            5,
            0.2,
            2,
            {"uc": 0, "analysis_window": 0.2},
            [(2, 30), (34, 76)],
        ),  # stereo_uc_0_analysis_window_0_2
        (
            0.2,
            5,
            0.2,
            2,
            {"uc": 1, "analysis_window": 0.2},
            [(10, 32), (36, 76)],
        ),  # stereo_uc_1_analysis_window_0_2
        (
            0.2,
            5,
            0,
            2,
            {"uc": "mix", "analysis_window": 0.1},
            [(10, 14), (17, 24), (26, 29), (36, 76)],
        ),  # stereo_uc_mix_aw_0_1_max_silence_0
        (
            0.2,
            5,
            0.1,
            2,
            {"uc": "mix", "analysis_window": 0.1},
            [(10, 15), (17, 25), (26, 30), (36, 76)],
        ),  # stereo_uc_mix_aw_0_1_max_silence_0_1
        (
            0.2,
            5,
            0.2,
            2,
            {"uc": "mix", "analysis_window": 0.1},
            [(10, 16), (17, 31), (36, 76)],
        ),  # stereo_uc_mix_aw_0_1_max_silence_0_2
        (
            0.2,
            5,
            0.3,
            2,
            {"uc": "mix", "analysis_window": 0.1},
            [(10, 32), (36, 76)],
        ),  # stereo_uc_mix_aw_0_1_max_silence_0_3
        (
            0.3,
            5,
            0,
            2,
            {"uc": "avg", "analysis_window": 0.2},
            [(10, 14), (16, 24), (36, 76)],
        ),  # stereo_uc_avg_aw_0_2_max_silence_0_min_dur_0_3
        (
            0.41,
            5,
            0,
            2,
            {"uc": "average", "analysis_window": 0.2},
            [(16, 24), (36, 76)],
        ),  # stereo_uc_average_aw_0_2_max_silence_0_min_dur_0_41
        (
            0.2,
            5,
            0.1,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 14), (16, 24), (26, 28), (36, 76)],
        ),  # stereo_uc_mix_aw_0_2_max_silence_0_1
        (
            0.2,
            5,
            0.2,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 30), (36, 76)],
        ),  # stereo_uc_mix_aw_0_2_max_silence_0_2
        (
            0.2,
            5,
            0.4,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 32), (36, 76)],
        ),  # stereo_uc_mix_aw_0_2_max_silence_0_4
        (
            0.2,
            5,
            0.5,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 32), (36, 76)],
        ),  # stereo_uc_mix_aw_0_2_max_silence_0_5
        (
            0.2,
            5,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 34), (36, 76)],
        ),  # stereo_uc_mix_aw_0_2_max_silence_0_6
        (
            0.2,
            5,
            0,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 24), (27, 30), (36, 76)],
        ),  # stereo_uc_mix_aw_0_3_max_silence_0
        (
            0.4,
            5,
            0,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 24), (36, 76)],
        ),  # stereo_uc_mix_aw_0_3_max_silence_0_min_dur_0_3
        (
            0.2,
            5,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 57), (57, 76)],
        ),  # stereo_uc_mix_aw_0_3_max_silence_0_6
        (
            0.2,
            5.1,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 60), (60, 76)],
        ),  # stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_1
        (
            0.2,
            5.2,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 60), (60, 76)],
        ),  # stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_2
        (
            0.2,
            5.3,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 60), (60, 76)],
        ),  # stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_3
        (
            0.2,
            5.4,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 63), (63, 76)],
        ),  # stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_4
        (
            0.2,
            5,
            0,
            2,
            {"uc": "mix", "analysis_window": 0.4},
            [(16, 24), (36, 76)],
        ),  # stereo_uc_mix_aw_0_4_max_silence_0
        (
            0.2,
            5,
            0.3,
            2,
            {"uc": "mix", "analysis_window": 0.4},
            [(16, 24), (36, 76)],
        ),  # stereo_uc_mix_aw_0_4_max_silence_0_3
        (
            0.2,
            5,
            0.4,
            2,
            {"uc": "mix", "analysis_window": 0.4},
            [(16, 28), (36, 76)],
        ),  # stereo_uc_mix_aw_0_4_max_silence_0_4
    ],
    ids=[
        "mono_aw_0_2_max_silence_0_2",
        "mono_aw_0_2_max_silence_0_3",
        "mono_aw_0_2_max_silence_0_4",
        "mono_aw_0_2_max_silence_0",
        "mono_aw_0_2",
        "mono_aw_0_3_max_silence_0",
        "mono_aw_0_3_max_silence_0_3",
        "mono_aw_0_3_max_silence_0_5",
        "mono_aw_0_3_max_silence_0_6",
        "mono_aw_0_4_max_silence_0",
        "mono_aw_0_4_max_silence_0_3",
        "mono_aw_0_4_max_silence_0_4",
        "stereo_uc_None_analysis_window_0_2",
        "stereo_uc_any_analysis_window_0_2",
        "stereo_use_channel_None_aw_0_3_max_silence_0_2",
        "stereo_use_channel_any_aw_0_3_max_silence_0_3",
        "stereo_use_channel_None_aw_0_4_max_silence_0_2",
        "stereo_use_channel_any_aw_0_3_max_silence_0_4",
        "stereo_uc_0_analysis_window_0_2",
        "stereo_uc_1_analysis_window_0_2",
        "stereo_uc_mix_aw_0_1_max_silence_0",
        "stereo_uc_mix_aw_0_1_max_silence_0_1",
        "stereo_uc_mix_aw_0_1_max_silence_0_2",
        "stereo_uc_mix_aw_0_1_max_silence_0_3",
        "stereo_uc_avg_aw_0_2_max_silence_0_min_dur_0_3",
        "stereo_uc_average_aw_0_2_max_silence_0_min_dur_0_41",
        "stereo_uc_mix_aw_0_2_max_silence_0_1",
        "stereo_uc_mix_aw_0_2_max_silence_0_2",
        "stereo_uc_mix_aw_0_2_max_silence_0_4",
        "stereo_uc_mix_aw_0_2_max_silence_0_5",
        "stereo_uc_mix_aw_0_2_max_silence_0_6",
        "stereo_uc_mix_aw_0_3_max_silence_0",
        "stereo_uc_mix_aw_0_3_max_silence_0_min_dur_0_3",
        "stereo_uc_mix_aw_0_3_max_silence_0_6",
        "stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_1",
        "stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_2",
        "stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_3",
        "stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_4",
        "stereo_uc_mix_aw_0_4_max_silence_0",
        "stereo_uc_mix_aw_0_4_max_silence_0_3",
        "stereo_uc_mix_aw_0_4_max_silence_0_4",
    ],
)
def test_split_analysis_window(
    min_dur, max_dur, max_silence, channels, kwargs, expected
):

    mono_or_stereo = "mono" if channels == 1 else "stereo"
    filename = "tests/data/test_split_10HZ_{}.raw".format(mono_or_stereo)
    with open(filename, "rb") as fp:
        data = fp.read()

    regions = split(
        data,
        min_dur=min_dur,
        max_dur=max_dur,
        max_silence=max_silence,
        drop_trailing_silence=False,
        strict_min_dur=False,
        sr=10,
        sw=2,
        ch=channels,
        eth=49.99,
        **kwargs
    )

    region = AudioRegion(data, 10, 2, channels)
    regions_ar = region.split(
        min_dur=min_dur,
        max_dur=max_dur,
        max_silence=max_silence,
        drop_trailing_silence=False,
        strict_min_dur=False,
        eth=49.99,
        **kwargs
    )

    regions = list(regions)
    regions_ar = list(regions_ar)
    err_msg = "Wrong number of regions after split, expected: "
    err_msg += "{}, found: {}".format(len(expected), len(regions))
    assert len(regions) == len(expected), err_msg
    err_msg = "Wrong number of regions after AudioRegion.split, expected: "
    err_msg += "{}, found: {}".format(len(expected), len(regions_ar))
    assert len(regions_ar) == len(expected), err_msg

    sample_width = 2
    sample_size_bytes = sample_width * channels
    for reg, reg_ar, exp in zip(regions, regions_ar, expected, strict=True):
        onset, offset = exp
        exp_data = data[onset * sample_size_bytes : offset * sample_size_bytes]
        assert bytes(reg) == exp_data
        assert reg == reg_ar


def test_split_custom_validator():
    filename = "tests/data/test_split_10HZ_mono.raw"
    with open(filename, "rb") as fp:
        data = fp.read()

    regions = split(
        data,
        min_dur=0.2,
        max_dur=5,
        max_silence=0.2,
        drop_trailing_silence=False,
        strict_min_dur=False,
        sr=10,
        sw=2,
        ch=1,
        analysis_window=0.1,
        validator=lambda x: to_array(x, sample_width=2, channels=1)[0] >= 320,
    )

    region = AudioRegion(data, 10, 2, 1)
    regions_ar = region.split(
        min_dur=0.2,
        max_dur=5,
        max_silence=0.2,
        drop_trailing_silence=False,
        strict_min_dur=False,
        analysis_window=0.1,
        validator=lambda x: to_array(x, sample_width=2, channels=1)[0] >= 320,
    )

    expected = [(2, 16), (17, 31), (34, 76)]
    regions = list(regions)
    regions_ar = list(regions_ar)
    err_msg = "Wrong number of regions after split, expected: "
    err_msg += "{}, found: {}".format(len(expected), len(regions))
    assert len(regions) == len(expected), err_msg
    err_msg = "Wrong number of regions after AudioRegion.split, expected: "
    err_msg += "{}, found: {}".format(len(expected), len(regions_ar))
    assert len(regions_ar) == len(expected), err_msg

    sample_size_bytes = 2
    for reg, reg_ar, exp in zip(regions, regions_ar, expected, strict=True):
        onset, offset = exp
        exp_data = data[onset * sample_size_bytes : offset * sample_size_bytes]
        assert bytes(reg) == exp_data
        assert reg == reg_ar


@pytest.mark.parametrize(
    "input, kwargs",
    [
        (
            "tests/data/test_split_10HZ_stereo.raw",
            {"audio_format": "raw", "sr": 10, "sw": 2, "ch": 2},
        ),  # filename_audio_format
        (
            "tests/data/test_split_10HZ_stereo.raw",
            {"fmt": "raw", "sr": 10, "sw": 2, "ch": 2},
        ),  # filename_audio_format_short_name
        (
            "tests/data/test_split_10HZ_stereo.raw",
            {"sr": 10, "sw": 2, "ch": 2},
        ),  # filename_no_audio_format
        (
            "tests/data/test_split_10HZ_stereo.raw",
            {"sampling_rate": 10, "sample_width": 2, "channels": 2},
        ),  # filename_no_long_audio_params
        (
            open("tests/data/test_split_10HZ_stereo.raw", "rb").read(),
            {"sr": 10, "sw": 2, "ch": 2},
        ),  # bytes_
        (
            AudioReader(
                "tests/data/test_split_10HZ_stereo.raw",
                sr=10,
                sw=2,
                ch=2,
                block_dur=0.1,
            ),
            {},
        ),  # audio_reader
        (
            AudioRegion(
                open("tests/data/test_split_10HZ_stereo.raw", "rb").read(),
                10,
                2,
                2,
            ),
            {},
        ),  # audio_region
        (
            get_audio_source(
                "tests/data/test_split_10HZ_stereo.raw", sr=10, sw=2, ch=2
            ),
            {},
        ),  # audio_source
    ],
    ids=[
        "filename_audio_format",
        "filename_audio_format_short_name",
        "filename_no_audio_format",
        "filename_no_long_audio_params",
        "bytes_",
        "audio_reader",
        "audio_region",
        "audio_source",
    ],
)
def test_split_input_type(input, kwargs):

    with open("tests/data/test_split_10HZ_stereo.raw", "rb") as fp:
        data = fp.read()

    regions = split(
        input,
        min_dur=0.2,
        max_dur=5,
        max_silence=0.2,
        drop_trailing_silence=False,
        strict_min_dur=False,
        analysis_window=0.1,
        **kwargs
    )
    regions = list(regions)
    expected = [(2, 32), (34, 76)]
    sample_width = 2
    err_msg = "Wrong number of regions after split, expected: "
    err_msg += "{}, found: {}".format(expected, regions)
    assert len(regions) == len(expected), err_msg
    for reg, exp in zip(regions, expected, strict=True):
        onset, offset = exp
        exp_data = data[onset * sample_width * 2 : offset * sample_width * 2]
        assert bytes(reg) == exp_data


@pytest.mark.parametrize(
    "min_dur, max_dur, analysis_window",
    [
        (0.5, 0.4, 0.1),
        (0.44, 0.49, 0.1),
    ],
    ids=[
        "min_dur_greater_than_max_dur",
        "durations_OK_but_wrong_number_of_analysis_windows",
    ],
)
def test_split_wrong_min_max_dur(min_dur, max_dur, analysis_window):

    with pytest.raises(ValueError) as val_err:
        split(
            b"0" * 16,
            min_dur=min_dur,
            max_dur=max_dur,
            max_silence=0.2,
            sr=16000,
            sw=1,
            ch=1,
            analysis_window=analysis_window,
        )

    err_msg = "'min_dur' ({0} sec.) results in {1} analysis "
    err_msg += "window(s) ({1} == ceil({0} / {2})) which is "
    err_msg += "higher than the number of analysis window(s) for "
    err_msg += "'max_dur' ({3} == floor({4} / {2}))"

    err_msg = err_msg.format(
        min_dur,
        math.ceil(min_dur / analysis_window),
        analysis_window,
        math.floor(max_dur / analysis_window),
        max_dur,
    )
    assert err_msg == str(val_err.value)


@pytest.mark.parametrize(
    "max_silence, max_dur, analysis_window",
    [
        (0.5, 0.5, 0.1),  # max_silence_equals_max_dur
        (0.5, 0.4, 0.1),  # max_silence_greater_than_max_dur
        (0.44, 0.49, 0.1),  # durations_OK_but_wrong_number_of_analysis_windows
    ],
    ids=[
        "max_silence_equals_max_dur",
        "max_silence_greater_than_max_dur",
        "durations_OK_but_wrong_number_of_analysis_windows",
    ],
)
def test_split_wrong_max_silence_max_dur(max_silence, max_dur, analysis_window):

    with pytest.raises(ValueError) as val_err:
        split(
            b"0" * 16,
            min_dur=0.2,
            max_dur=max_dur,
            max_silence=max_silence,
            sr=16000,
            sw=1,
            ch=1,
            analysis_window=analysis_window,
        )

    err_msg = "'max_silence' ({0} sec.) results in {1} analysis "
    err_msg += "window(s) ({1} == floor({0} / {2})) which is "
    err_msg += "higher or equal to the number of analysis window(s) for "
    err_msg += "'max_dur' ({3} == floor({4} / {2}))"

    err_msg = err_msg.format(
        max_silence,
        math.floor(max_silence / analysis_window),
        analysis_window,
        math.floor(max_dur / analysis_window),
        max_dur,
    )
    assert err_msg == str(val_err.value)


@pytest.mark.parametrize(
    "wrong_param",
    [
        {"min_dur": -1},  # negative_min_dur
        {"min_dur": 0},  # zero_min_dur
        {"max_dur": -1},  # negative_max_dur
        {"max_dur": 0},  # zero_max_dur
        {"max_silence": -1},  # negative_max_silence
        {"analysis_window": 0},  # zero_analysis_window
        {"analysis_window": -1},  # negative_analysis_window
    ],
    ids=[
        "negative_min_dur",
        "zero_min_dur",
        "negative_max_dur",
        "zero_max_dur",
        "negative_max_silence",
        "zero_analysis_window",
        "negative_analysis_window",
    ],
)
def test_split_negative_temporal_params(wrong_param):

    params = {
        "min_dur": 0.2,
        "max_dur": 0.5,
        "max_silence": 0.1,
        "analysis_window": 0.1,
    }
    params.update(wrong_param)
    with pytest.raises(ValueError) as val_err:
        split(None, **params)

    name = set(wrong_param).pop()
    value = wrong_param[name]
    err_msg = "'{}' ({}) must be >{} 0".format(
        name, value, "=" if name == "max_silence" else ""
    )
    assert err_msg == str(val_err.value)


def test_split_too_small_analysis_window():
    with pytest.raises(ValueError) as val_err:
        split(b"", sr=10, sw=1, ch=1, analysis_window=0.09)
    err_msg = "Too small 'analysis_window' (0.09) for sampling rate (10)."
    err_msg += " Analysis window should at least be 1/10 to cover one "
    err_msg += "data sample"
    assert err_msg == str(val_err.value)


def test_split_and_plot():

    with open("tests/data/test_split_10HZ_mono.raw", "rb") as fp:
        data = fp.read()

    region = AudioRegion(data, 10, 2, 1)
    with patch("auditok.core.plot") as patch_fn:
        regions = region.split_and_plot(
            min_dur=0.2,
            max_dur=5,
            max_silence=0.2,
            drop_trailing_silence=False,
            strict_min_dur=False,
            analysis_window=0.1,
            sr=10,
            sw=2,
            ch=1,
            eth=50,
        )
    assert patch_fn.called
    expected = [(2, 16), (17, 31), (34, 76)]
    sample_width = 2
    expected_regions = []
    for onset, offset in expected:
        onset *= sample_width
        offset *= sample_width
        expected_regions.append(AudioRegion(data[onset:offset], 10, 2, 1))
    assert regions == expected_regions


def test_split_exception():
    with open("tests/data/test_split_10HZ_mono.raw", "rb") as fp:
        data = fp.read()
        region = AudioRegion(data, 10, 2, 1)

    with pytest.raises(RuntimeWarning):
        # max_read is not accepted when calling AudioRegion.split
        region.split(max_read=2)


@pytest.mark.parametrize(
    (
        "data, start, sampling_rate, sample_width, channels, expected_end, "
        + "expected_duration_s, expected_duration_ms"
    ),
    [
        (b"\0" * 8000, 0, 8000, 1, 1, 1, 1, 1000),  # simple
        (
            b"\0" * 7992,
            0,
            8000,
            1,
            1,
            0.999,
            0.999,
            999,
        ),  # one_ms_less_than_1_sec
        (
            b"\0" * 7994,
            0,
            8000,
            1,
            1,
            0.99925,
            0.99925,
            999,
        ),  # tree_quarter_ms_less_than_1_sec
        (
            b"\0" * 7996,
            0,
            8000,
            1,
            1,
            0.9995,
            0.9995,
            1000,
        ),  # half_ms_less_than_1_sec
        (
            b"\0" * 7998,
            0,
            8000,
            1,
            1,
            0.99975,
            0.99975,
            1000,
        ),  # quarter_ms_less_than_1_sec
        (b"\0" * 8000 * 2, 0, 8000, 2, 1, 1, 1, 1000),  # simple_sample_width_2
        (b"\0" * 8000 * 2, 0, 8000, 1, 2, 1, 1, 1000),  # simple_stereo
        (b"\0" * 8000 * 5, 0, 8000, 1, 5, 1, 1, 1000),  # simple_multichannel
        (
            b"\0" * 8000 * 2 * 5,
            0,
            8000,
            2,
            5,
            1,
            1,
            1000,
        ),  # simple_sample_width_2_multichannel
        (
            b"\0" * 7992 * 2 * 5,
            0,
            8000,
            2,
            5,
            0.999,
            0.999,
            999,
        ),  # one_ms_less_than_1s_sw_2_multichannel
        (
            b"\0" * 7994 * 2 * 5,
            0,
            8000,
            2,
            5,
            0.99925,
            0.99925,
            999,
        ),  # tree_qrt_ms_lt_1_s_sw_2_multichannel
        (
            b"\0" * 7996 * 2 * 5,
            0,
            8000,
            2,
            5,
            0.9995,
            0.9995,
            1000,
        ),  # half_ms_lt_1s_sw_2_multichannel
        (
            b"\0" * 7998 * 2 * 5,
            0,
            8000,
            2,
            5,
            0.99975,
            0.99975,
            1000,
        ),  # quarter_ms_lt_1s_sw_2_multichannel
        (
            b"\0" * int(8000 * 1.33),
            2.7,
            8000,
            1,
            1,
            4.03,
            1.33,
            1330,
        ),  # arbitrary_length_1
        (
            b"\0" * int(8000 * 0.476),
            11.568,
            8000,
            1,
            1,
            12.044,
            0.476,
            476,
        ),  # arbitrary_length_2
        (
            b"\0" * int(8000 * 1.711) * 2 * 3,
            9.415,
            8000,
            2,
            3,
            11.126,
            1.711,
            1711,
        ),  # arbitrary_length_sw_2_multichannel
        (
            b"\0" * int(3172 * 1.318),
            17.236,
            3172,
            1,
            1,
            17.236 + int(3172 * 1.318) / 3172,
            int(3172 * 1.318) / 3172,
            1318,
        ),  # arbitrary_sampling_rate
        (
            b"\0" * int(11317 * 0.716) * 2 * 3,
            18.811,
            11317,
            2,
            3,
            18.811 + int(11317 * 0.716) / 11317,
            int(11317 * 0.716) / 11317,
            716,
        ),  # arbitrary_sr_sw_2_multichannel
    ],
    ids=[
        "simple",
        "one_ms_less_than_1_sec",
        "tree_quarter_ms_less_than_1_sec",
        "half_ms_less_than_1_sec",
        "quarter_ms_less_than_1_sec",
        "simple_sample_width_2",
        "simple_stereo",
        "simple_multichannel",
        "simple_sample_width_2_multichannel",
        "one_ms_less_than_1s_sw_2_multichannel",
        "tree_qrt_ms_lt_1_s_sw_2_multichannel",
        "half_ms_lt_1s_sw_2_multichannel",
        "quarter_ms_lt_1s_sw_2_multichannel",
        "arbitrary_length_1",
        "arbitrary_length_2",
        "arbitrary_length_sw_2_multichannel",
        "arbitrary_sampling_rate",
        "arbitrary_sr_sw_2_multichannel",
    ],
)
def test_creation(
    data,
    start,
    sampling_rate,
    sample_width,
    channels,
    expected_end,
    expected_duration_s,
    expected_duration_ms,
):
    region = AudioRegion(data, sampling_rate, sample_width, channels, start)
    assert region.sampling_rate == sampling_rate
    assert region.sr == sampling_rate
    assert region.sample_width == sample_width
    assert region.sw == sample_width
    assert region.channels == channels
    assert region.ch == channels
    assert region.meta.start == start
    assert region.meta.end == expected_end
    assert region.duration == expected_duration_s
    assert len(region.ms) == expected_duration_ms
    assert bytes(region) == data


def test_creation_invalid_data_exception():
    with pytest.raises(AudioParameterError) as audio_param_err:
        _ = AudioRegion(
            data=b"ABCDEFGHI", sampling_rate=8, sample_width=2, channels=1
        )
    assert str(audio_param_err.value) == (
        "The length of audio data must be an integer "
        "multiple of `sample_width * channels`"
    )


@pytest.mark.parametrize(
    "skip, max_read, channels",
    [
        (0, -1, 1),  # no_skip_read_all
        (0, -1, 2),  # no_skip_read_all_stereo
        (2, -1, 1),  # skip_2_read_all
        (2, None, 1),  # skip_2_read_all_None
        (2, 3, 1),  # skip_2_read_3
        (2, 3.5, 2),  # skip_2_read_3_5_stereo
        (2.4, 3.5, 2),  # skip_2_4_read_3_5_stereo
    ],
    ids=[
        "no_skip_read_all",
        "no_skip_read_all_stereo",
        "skip_2_read_all",
        "skip_2_read_all_None",
        "skip_2_read_3",
        "skip_2_read_3_5_stereo",
        "skip_2_4_read_3_5_stereo",
    ],
)
def test_load_AudioRegion(skip, max_read, channels):
    sampling_rate = 10
    sample_width = 2
    filename = "tests/data/test_split_10HZ_{}.raw"
    filename = filename.format("mono" if channels == 1 else "stereo")
    region = AudioRegion.load(
        filename,
        skip=skip,
        max_read=max_read,
        sr=sampling_rate,
        sw=sample_width,
        ch=channels,
    )
    with open(filename, "rb") as fp:
        fp.read(round(skip * sampling_rate * sample_width * channels))
        if max_read is None or max_read < 0:
            to_read = -1
        else:
            to_read = round(max_read * sampling_rate * sample_width * channels)
        expected = fp.read(to_read)
    assert bytes(region) == expected


def test_load_from_microphone():
    with patch("auditok.io.PyAudioSource") as patch_pyaudio_source:
        with patch("auditok.core.AudioReader.read") as patch_reader:
            patch_reader.return_value = None
            with patch(
                "auditok.core.AudioRegion.__init__"
            ) as patch_AudioRegion:
                patch_AudioRegion.return_value = None
                AudioRegion.load(None, skip=0, max_read=5, sr=16000, sw=2, ch=1)
    assert patch_pyaudio_source.called
    assert patch_reader.called
    assert patch_AudioRegion.called


@pytest.mark.parametrize(
    "max_read",
    [
        None,  # None
        -1,  # negative
    ],
    ids=[
        "None",
        "negative",
    ],
)
def test_load_from_microphone_without_max_read_exception(max_read):
    with pytest.raises(ValueError) as val_err:
        AudioRegion.load(None, max_read=max_read, sr=16000, sw=2, ch=1)
    assert str(val_err.value) == (
        "'max_read' should not be None when reading from microphone"
    )


def test_load_from_microphone_with_nonzero_skip_exception():
    with pytest.raises(ValueError) as val_err:
        AudioRegion.load(None, skip=1, max_read=5, sr=16000, sw=2, ch=1)
    assert str(val_err.value) == (
        "'skip' should be 0 when reading from microphone"
    )


@pytest.mark.parametrize(
    "format, start, expected",
    [
        ("output.wav", 1.230, "output.wav"),  # simple
        ("output_{meta.start:g}.wav", 1.230, "output_1.23.wav"),  # start
        ("output_{meta.start}.wav", 1.233712, "output_1.233712.wav"),  # start_2
        (
            "output_{meta.start:.2f}.wav",
            1.2300001,
            "output_1.23.wav",
        ),  # start_3
        (
            "output_{meta.start:.3f}.wav",
            1.233712,
            "output_1.234.wav",
        ),  # start_4
        (
            "output_{meta.start:.8f}.wav",
            1.233712,
            "output_1.23371200.wav",
        ),  # start_5
        (
            "output_{meta.start}_{meta.end}_{duration}.wav",
            1.455,
            "output_1.455_2.455_1.0.wav",
        ),  # start_end_duration
        (
            "output_{meta.start}_{meta.end}_{duration}.wav",
            1.455321,
            "output_1.455321_2.455321_1.0.wav",
        ),  # start_end_duration_2
    ],
    ids=[
        "simple",
        "start",
        "start_2",
        "start_3",
        "start_4",
        "start_5",
        "start_end_duration",
        "start_end_duration_2",
    ],
)
def test_save(format, start, expected):
    with TemporaryDirectory() as tmpdir:
        region = AudioRegion(b"0" * 160, 160, 1, 1, start)
        format = os.path.join(tmpdir, format)
        filename = region.save(format)[len(tmpdir) + 1 :]
        assert filename == expected


def test_save_file_exists_exception():
    with TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "output.wav")
        open(filename, "w").close()
        region = AudioRegion(b"0" * 160, 160, 1, 1)
        with pytest.raises(FileExistsError):
            region.save(filename, exists_ok=False)

        with pytest.raises(FileExistsError):
            region.save(Path(filename), exists_ok=False)


@pytest.mark.parametrize(
    "sampling_rate, sample_width, channels",
    [
        (16000, 1, 1),  # mono_16K_1byte
        (16000, 2, 1),  # mono_16K_2byte
        (44100, 2, 2),  # stereo_44100_2byte
        (44100, 2, 3),  # 3channel_44100_2byte
    ],
    ids=[
        "mono_16K_1byte",
        "mono_16K_2byte",
        "stereo_44100_2byte",
        "3channel_44100_2byte",
    ],
)
def test_join(sampling_rate, sample_width, channels):
    duration = 1
    size = int(duration * sampling_rate * sample_width * channels)
    glue_data = b"\0" * size
    regions_data = [
        b"\1" * int(size * 1.5),
        b"\2" * int(size * 0.5),
        b"\3" * int(size * 0.75),
    ]

    glue_region = AudioRegion(glue_data, sampling_rate, sample_width, channels)
    regions = [
        AudioRegion(data, sampling_rate, sample_width, channels)
        for data in regions_data
    ]
    joined = glue_region.join(regions)
    assert joined.data == glue_data.join(regions_data)
    assert joined.duration == duration * 2 + 1.5 + 0.5 + 0.75


@pytest.mark.parametrize(
    "sampling_rate, sample_width, channels",
    [
        (32000, 1, 1),  # different_sampling_rate
        (16000, 2, 1),  # different_sample_width
        (16000, 1, 2),  # different_channels
    ],
    ids=[
        "different_sampling_rate",
        "different_sample_width",
        "different_channels",
    ],
)
def test_join_exception(sampling_rate, sample_width, channels):

    glue_sampling_rate = 16000
    glue_sample_width = 1
    glue_channels = 1

    duration = 1
    size = int(
        duration * glue_sampling_rate * glue_sample_width * glue_channels
    )
    glue_data = b"\0" * size
    glue_region = AudioRegion(
        glue_data, glue_sampling_rate, glue_sample_width, glue_channels
    )

    size = int(duration * sampling_rate * sample_width * channels)
    regions_data = [
        b"\1" * int(size * 1.5),
        b"\2" * int(size * 0.5),
        b"\3" * int(size * 0.75),
    ]
    regions = [
        AudioRegion(data, sampling_rate, sample_width, channels)
        for data in regions_data
    ]

    with pytest.raises(AudioParameterError):
        glue_region.join(regions)


@pytest.mark.parametrize(
    "region, slice_, expected_data",
    [
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(0, 500),
            b"a" * 80,  # first_half
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(500, None),
            b"b" * 80,  # second_half
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-500, None),
            b"b" * 80,  # second_half_negative
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(200, 750),
            b"a" * 48 + b"b" * 40,  # middle
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-800, -250),
            b"a" * 48 + b"b" * 40,  # middle_negative
        ),
        (
            AudioRegion(b"a" * 160 + b"b" * 160, 160, 2, 1),
            slice(200, 750),
            b"a" * 96 + b"b" * 80,  # middle_sw2
        ),
        (
            AudioRegion(b"a" * 160 + b"b" * 160, 160, 1, 2),
            slice(200, 750),
            b"a" * 96 + b"b" * 80,  # middle_ch2
        ),
        (
            AudioRegion(b"a" * 320 + b"b" * 320, 160, 2, 2),
            slice(200, 750),
            b"a" * 192 + b"b" * 160,  # middle_sw2_ch2
        ),
        (
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(1, None),
            b"a" * (4000 - 8) + b"b" * 4000,  # but_first_sample
        ),
        (
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(-999, None),
            b"a" * (4000 - 8) + b"b" * 4000,  # but_first_sample_negative
        ),
        (
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(0, 999),
            b"a" * 4000 + b"b" * (4000 - 8),  # but_last_sample
        ),
        (
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(0, -1),
            b"a" * 4000 + b"b" * (4000 - 8),  # but_last_sample_negative
        ),
        (
            AudioRegion(b"a" * 160, 160, 1, 1),
            slice(-5000, None),
            b"a" * 160,  # big_negative_start
        ),
        (
            AudioRegion(b"a" * 160, 160, 1, 1),
            slice(None, -1500),
            b"",  # big_negative_stop
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(0, 0),
            b"",  # empty
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(200, 100),
            b"",  # empty_start_stop_reversed
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(2000, 3000),
            b"",  # empty_big_positive_start
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-100, -200),
            b"",  # empty_negative_reversed
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(0, -2000),
            b"",  # empty_big_negative_stop
        ),
        (
            AudioRegion(b"a" * 124 + b"b" * 376, 1234, 1, 1),
            slice(100, 200),
            b"a" + b"b" * 123,  # arbitrary_sampling_rate
        ),
    ],
    ids=[
        "first_half",
        "second_half",
        "second_half_negative",
        "middle",
        "middle_negative",
        "middle_sw2",
        "middle_ch2",
        "middle_sw2_ch2",
        "but_first_sample",
        "but_first_sample_negative",
        "but_last_sample",
        "but_last_sample_negative",
        "big_negative_start",
        "big_negative_stop",
        "empty",
        "empty_start_stop_reversed",
        "empty_big_positive_start",
        "empty_negative_reversed",
        "empty_big_negative_stop",
        "arbitrary_sampling_rate",
    ],
)
def test_region_temporal_slicing(region, slice_, expected_data):
    sub_region = region.millis[slice_]
    assert bytes(sub_region) == expected_data
    start_sec = slice_.start / 1000 if slice_.start is not None else None
    stop_sec = slice_.stop / 1000 if slice_.stop is not None else None
    sub_region = region.sec[start_sec:stop_sec]
    assert bytes(sub_region) == expected_data


@pytest.mark.parametrize(
    "region, slice_, time_shift, expected_data",
    [
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(0, 80),
            0,
            b"a" * 80,  # first_half
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(80, None),
            0.5,
            b"b" * 80,  # second_half
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-80, None),
            0.5,
            b"b" * 80,  # second_half_negative
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(160 // 5, 160 // 4 * 3),
            0.2,
            b"a" * 48 + b"b" * 40,  # middle
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-160 // 5 * 4, -160 // 4),
            0.2,
            b"a" * 48 + b"b" * 40,  # middle_negative
        ),
        (
            AudioRegion(b"a" * 160 + b"b" * 160, 160, 2, 1),
            slice(160 // 5, 160 // 4 * 3),
            0.2,
            b"a" * 96 + b"b" * 80,  # middle_sw2
        ),
        (
            AudioRegion(b"a" * 160 + b"b" * 160, 160, 1, 2),
            slice(160 // 5, 160 // 4 * 3),
            0.2,
            b"a" * 96 + b"b" * 80,  # middle_ch2
        ),
        (
            AudioRegion(b"a" * 320 + b"b" * 320, 160, 2, 2),
            slice(160 // 5, 160 // 4 * 3),
            0.2,
            b"a" * 192 + b"b" * 160,  # middle_sw2_ch2
        ),
        (
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(1, None),
            1 / 8000,
            b"a" * (4000 - 1) + b"b" * 4000,  # but_first_sample
        ),
        (
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(-7999, None),
            1 / 8000,
            b"a" * (4000 - 1) + b"b" * 4000,  # but_first_sample_negative
        ),
        (
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(0, 7999),
            0,
            b"a" * 4000 + b"b" * (4000 - 1),  # but_last_sample
        ),
        (
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(0, -1),
            0,
            b"a" * 4000 + b"b" * (4000 - 1),  # but_last_sample_negative
        ),
        (
            AudioRegion(b"a" * 160, 160, 1, 1),
            slice(-1600, None),
            0,
            b"a" * 160,  # big_negative_start
        ),
        (
            AudioRegion(b"a" * 160, 160, 1, 1),
            slice(None, -1600),
            0,
            b"",  # big_negative_stop
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(0, 0),
            0,
            b"",  # empty
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(80, 40),
            0.5,
            b"",  # empty_start_stop_reversed
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(1600, 3000),
            10,
            b"",  # empty_big_positive_start
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-16, -32),
            0.9,
            b"",  # empty_negative_reversed
        ),
        (
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(0, -2000),
            0,
            b"",  # empty_big_negative_stop
        ),
        (
            AudioRegion(b"a" * 124 + b"b" * 376, 1235, 1, 1),
            slice(100, 200),
            100 / 1235,
            b"a" * 24 + b"b" * 76,  # arbitrary_sampling_rate
        ),
        (
            AudioRegion(b"a" * 124 + b"b" * 376, 1235, 2, 2),
            slice(25, 50),
            25 / 1235,
            b"a" * 24 + b"b" * 76,  # arbitrary_sampling_rate_middle_sw2_ch2
        ),
    ],
    ids=[
        "first_half",
        "second_half",
        "second_half_negative",
        "middle",
        "middle_negative",
        "middle_sw2",
        "middle_ch2",
        "middle_sw2_ch2",
        "but_first_sample",
        "but_first_sample_negative",
        "but_last_sample",
        "but_last_sample_negative",
        "big_negative_start",
        "big_negative_stop",
        "empty",
        "empty_start_stop_reversed",
        "empty_big_positive_start",
        "empty_negative_reversed",
        "empty_big_negative_stop",
        "arbitrary_sampling_rate",
        "arbitrary_sampling_rate_middle_sw2_ch2",
    ],
)
def test_region_sample_slicing(region, slice_, time_shift, expected_data):
    sub_region = region[slice_]
    assert bytes(sub_region) == expected_data


@pytest.mark.parametrize(
    "sampling_rate, sample_width, channels",
    [
        (8000, 1, 1),  # simple
        (8000, 2, 2),  # stereo_sw_2
        (5413, 2, 3),  # arbitrary_sr_multichannel
    ],
    ids=[
        "simple",
        "stereo_sw_2",
        "arbitrary_sr_multichannel",
    ],
)
def test_concatenation(sampling_rate, sample_width, channels):

    region_1, region_2 = _make_random_length_regions(
        [b"a", b"b"], sampling_rate, sample_width, channels
    )
    expected_duration = region_1.duration + region_2.duration
    expected_data = bytes(region_1) + bytes(region_2)
    concat_region = region_1 + region_2
    assert concat_region.duration == pytest.approx(expected_duration, abs=1e-6)
    assert bytes(concat_region) == expected_data


@pytest.mark.parametrize(
    "sampling_rate, sample_width, channels",
    [
        (8000, 1, 1),  # simple
        (8000, 2, 2),  # stereo_sw_2
        (5413, 2, 3),  # arbitrary_sr_multichannel
    ],
    ids=[
        "simple",
        "stereo_sw_2",
        "arbitrary_sr_multichannel",
    ],
)
def test_concatenation_many(sampling_rate, sample_width, channels):

    regions = _make_random_length_regions(
        [b"a", b"b", b"c"], sampling_rate, sample_width, channels
    )
    expected_duration = sum(r.duration for r in regions)
    expected_data = b"".join(bytes(r) for r in regions)
    concat_region = sum(regions)

    assert concat_region.duration == pytest.approx(expected_duration, abs=1e-6)
    assert bytes(concat_region) == expected_data


def test_concatenation_different_sampling_rate_error():
    region_1 = AudioRegion(b"a" * 100, 8000, 1, 1)
    region_2 = AudioRegion(b"b" * 100, 3000, 1, 1)

    with pytest.raises(AudioParameterError) as val_err:
        region_1 + region_2
    assert str(val_err.value) == (
        "Can only concatenate AudioRegions of the same "
        "sampling rate (8000 != 3000)"  # different_sampling_rate
    )


def test_concatenation_different_sample_width_error():
    region_1 = AudioRegion(b"a" * 100, 8000, 2, 1)
    region_2 = AudioRegion(b"b" * 100, 8000, 4, 1)

    with pytest.raises(AudioParameterError) as val_err:
        region_1 + region_2
    assert str(val_err.value) == (
        "Can only concatenate AudioRegions of the same sample width (2 != 4)"
    )


def test_concatenation_different_number_of_channels_error():
    region_1 = AudioRegion(b"a" * 100, 8000, 1, 1)
    region_2 = AudioRegion(b"b" * 100, 8000, 1, 2)

    with pytest.raises(AudioParameterError) as val_err:
        region_1 + region_2
    assert str(val_err.value) == (
        "Can only concatenate AudioRegions of the same "
        "number of channels (1 != 2)"  # different_number_of_channels
    )


@pytest.mark.parametrize(
    "duration, expected_duration, expected_len, expected_len_ms",
    [
        (0.01, 0.03, 240, 30),  # simple
        (0.00575, 0.01725, 138, 17),  # rounded_len_floor
        (0.00625, 0.01875, 150, 19),  # rounded_len_ceil
    ],
    ids=[
        "simple",
        "rounded_len_floor",
        "rounded_len_ceil",
    ],
)
def test_multiplication(
    duration, expected_duration, expected_len, expected_len_ms
):
    sw = 2
    data = b"0" * int(duration * 8000 * sw)
    region = AudioRegion(data, 8000, sw, 1)
    m_region = 1 * region * 3
    assert bytes(m_region) == data * 3
    assert m_region.sr == 8000
    assert m_region.sw == 2
    assert m_region.ch == 1
    assert m_region.duration == expected_duration
    assert len(m_region) == expected_len
    assert m_region.len == expected_len
    assert m_region.s.len == expected_duration
    assert len(m_region.ms) == expected_len_ms
    assert m_region.ms.len == expected_len_ms


@pytest.mark.parametrize(
    "factor, _type",
    [
        ("x", str),  # string
        (1.4, float),  # float
    ],
    ids=[
        "string",
        "float",
    ],
)
def test_multiplication_non_int(factor, _type):
    with pytest.raises(TypeError) as type_err:
        AudioRegion(b"0" * 80, 8000, 1, 1) * factor
    err_msg = "Can't multiply AudioRegion by a non-int of type '{}'"
    assert err_msg.format(_type) == str(type_err.value)


@pytest.mark.parametrize(
    "data",
    [
        [b"a" * 80, b"b" * 80],  # simple
        [b"a" * 31, b"b" * 31, b"c" * 30],  # extra_samples_1
        [b"a" * 31, b"b" * 30, b"c" * 30],  # extra_samples_2
        [b"a" * 11, b"b" * 11, b"c" * 10, b"c" * 10],  # extra_samples_3
    ],
    ids=[
        "simple",
        "extra_samples_1",
        "extra_samples_2",
        "extra_samples_3",
    ],
)
def test_truediv(data):

    region = AudioRegion(b"".join(data), 80, 1, 1)

    sub_regions = region / len(data)
    for data_i, region in zip(data, sub_regions, strict=True):
        assert len(data_i) == len(bytes(region))


@pytest.mark.parametrize(
    "data, sample_width, channels, expected",
    [
        (b"a" * 10, 1, 1, [97] * 10),  # mono_sw_1
        (b"a" * 10, 2, 1, [24929] * 5),  # mono_sw_2
        (b"a" * 8, 4, 1, [1633771873] * 2),  # mono_sw_4
        (b"ab" * 5, 1, 2, [[97] * 5, [98] * 5]),  # stereo_sw_1
    ],
    ids=[
        "mono_sw_1",
        "mono_sw_2",
        "mono_sw_4",
        "stereo_sw_1",
    ],
)
def test_samples(data, sample_width, channels, expected):

    region = AudioRegion(data, 10, sample_width, channels)
    expected = np.array(expected)
    assert (region.samples == expected).all()
    assert (region.numpy() == expected).all()
    assert (np.array(region) == expected).all()
