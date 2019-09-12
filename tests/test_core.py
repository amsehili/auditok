import os
import math
from random import random
from tempfile import TemporaryDirectory
from array import array as array_
from unittest import TestCase
from mock import patch
from genty import genty, genty_dataset
from auditok import split, AudioRegion, AudioParameterError
from auditok.core import _duration_to_nb_windows
from auditok.util import AudioDataSource
from auditok.io import (
    _normalize_use_channel,
    _extract_selected_channel,
    get_audio_source,
)


def _make_random_length_regions(byte_seq, sampling_rate, sample_width, channels):
    regions = []
    for b in byte_seq:
        duration = round(random() * 10, 6)
        data = b * int(duration * sampling_rate) * sample_width * channels
        region = AudioRegion(data, sampling_rate, sample_width, channels)
        regions.append(region)
    return regions


@genty
class TestFunctions(TestCase):
    @genty_dataset(
        zero_duration=(0, 1, None, 0),
        multiple=(0.3, 0.1, round, 3),
        not_multiple_ceil=(0.35, 0.1, math.ceil, 4),
        not_multiple_floor=(0.35, 0.1, math.floor, 3),
        small_duration=(0.05, 0.1, round, 0),
        small_duration_ceil=(0.05, 0.1, math.ceil, 1),
        with_round_error=(0.3, 0.1, math.floor, 3, {"epsilon": 1e-6}),
        negative_duration=(-0.5, 0.1, math.ceil, ValueError),
        negative_analysis_window=(0.5, -0.1, math.ceil, ValueError),
    )
    def test_duration_to_nb_windows(
        self, duration, analysis_window, round_fn, expected, kwargs=None
    ):
        if expected == ValueError:
            with self.assertRaises(expected):
                _duration_to_nb_windows(duration, analysis_window, round_fn)
        else:
            if kwargs is None:
                kwargs = {}
            result = _duration_to_nb_windows(
                duration, analysis_window, round_fn, **kwargs
            )
            self.assertEqual(result, expected)


@genty
class TestSplit(TestCase):
    @genty_dataset(
        simple=(0.2, 5, 0.2, False, False, {"eth": 50}, [(2, 16), (17, 31), (34, 76)]),
        short_max_dur=(
            0.3,
            2,
            0.2,
            False,
            False,
            {"eth": 50},
            [(2, 16), (17, 31), (34, 54), (54, 74), (74, 76)],
        ),
        long_min_dur=(3, 5, 0.2, False, False, {"eth": 50}, [(34, 76)]),
        long_max_silence=(0.2, 80, 10, False, False, {"eth": 50}, [(2, 76)]),
        zero_max_silence=(
            0.2,
            5,
            0.0,
            False,
            False,
            {"eth": 50},
            [(2, 14), (17, 24), (26, 29), (34, 76)],
        ),
        low_energy_threshold=(
            0.2,
            5,
            0.2,
            False,
            False,
            {"energy_threshold": 40},
            [(0, 50), (50, 76)],
        ),
        high_energy_threshold=(0.2, 5, 0.2, False, False, {"energy_threshold": 60}, []),
        trim_leading_and_trailing_silence=(
            0.2,
            10,  # use long max_dur
            0.5,  # and a max_silence longer than any inter-region silence
            True,
            False,
            {"eth": 50},
            [(2, 76)],
        ),
        drop_trailing_silence=(
            0.2,
            5,
            0.2,
            True,
            False,
            {"eth": 50},
            [(2, 14), (17, 29), (34, 76)],
        ),
        drop_trailing_silence_2=(1.5, 5, 0.2, True, False, {"eth": 50}, [(34, 76)]),
        strict_min_dur=(
            0.3,
            2,
            0.2,
            False,
            True,
            {"eth": 50},
            [(2, 16), (17, 31), (34, 54), (54, 74)],
        ),
    )
    def test_split_params(
        self,
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
        self.assertEqual(len(regions), len(expected), err_msg)
        err_msg = "Wrong number of regions after AudioRegion.split, expected: "
        err_msg += "{}, found: {}".format(len(expected), len(regions_ar))
        self.assertEqual(len(regions_ar), len(expected), err_msg)

        sample_width = 2
        for reg, reg_ar, exp in zip(regions, regions_ar, expected):
            onset, offset = exp
            exp_data = data[onset * sample_width : offset * sample_width]
            self.assertEqual(bytes(reg), exp_data)
            self.assertEqual(reg, reg_ar)
  
    @genty_dataset(
        stereo_all_default=(2, {}, [(2, 32), (34, 76)]),
        mono_max_read=(1, {"max_read": 5}, [(2, 16), (17, 31), (34, 50)]),
        mono_max_read_short_name=(1, {"mr": 5}, [(2, 16), (17, 31), (34, 50)]),
        mono_use_channel_1=(
            1,
            {"eth": 50, "use_channel": 0},
            [(2, 16), (17, 31), (34, 76)],
        ),
        mono_uc_1=(1, {"eth": 50, "uc": 1}, [(2, 16), (17, 31), (34, 76)]),
        mono_use_channel_None=(
            1,
            {"eth": 50, "use_channel": None},
            [(2, 16), (17, 31), (34, 76)],
        ),
        stereo_use_channel_1=(
            2,
            {"eth": 50, "use_channel": 0},
            [(2, 16), (17, 31), (34, 76)],
        ),
        stereo_use_channel_no_use_channel_given=(2, {"eth": 50}, [(2, 32), (34, 76)]),
        stereo_use_channel_minus_2=(
            2,
            {"eth": 50, "use_channel": -2},
            [(2, 16), (17, 31), (34, 76)],
        ),
        stereo_uc_2=(2, {"eth": 50, "uc": 1}, [(10, 32), (36, 76)]),
        stereo_uc_minus_1=(2, {"eth": 50, "uc": -1}, [(10, 32), (36, 76)]),
        mono_uc_mix=(1, {"eth": 50, "uc": "mix"}, [(2, 16), (17, 31), (34, 76)]),
        stereo_use_channel_mix=(
            2,
            {"energy_threshold": 53.5, "use_channel": "mix"},
            [(54, 76)],
        ),
        stereo_uc_mix=(2, {"eth": 52, "uc": "mix"}, [(17, 26), (54, 76)]),
        stereo_uc_mix_default_eth=(2, {"uc": "mix"}, [(10, 16), (17, 31), (36, 76)]),
    )
    def test_split_kwargs(self, channels, kwargs, expected):

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
        self.assertEqual(len(regions), len(expected), err_msg)
        err_msg = "Wrong number of regions after AudioRegion.split, expected: "
        err_msg += "{}, found: {}".format(len(expected), len(regions_ar))
        self.assertEqual(len(regions_ar), len(expected), err_msg)

        sample_width = 2
        sample_size_bytes = sample_width * channels
        for reg, reg_ar, exp in zip(regions, regions_ar, expected):
            onset, offset = exp
            exp_data = data[onset * sample_size_bytes : offset * sample_size_bytes]
            self.assertEqual(len(bytes(reg)), len(exp_data))
            self.assertEqual(reg, reg_ar)

    @genty_dataset(
        mono_aw_0_2_max_silence_0_2=(0.2, 5, 0.2, 1, {"aw": 0.2}, [(2, 30), (34, 76)]),
        mono_aw_0_2_max_silence_0_3=(0.2, 5, 0.3, 1, {"aw": 0.2}, [(2, 30), (34, 76)]),
        mono_aw_0_2_max_silence_0_4=(0.2, 5, 0.4, 1, {"aw": 0.2}, [(2, 32), (34, 76)]),
        mono_aw_0_2_max_silence_0=(
            0.2,
            5,
            0,
            1,
            {"aw": 0.2},
            [(2, 14), (16, 24), (26, 28), (34, 76)],
        ),
        mono_aw_0_2=(0.2, 5, 0.2, 1, {"aw": 0.2}, [(2, 30), (34, 76)]),
        mono_aw_0_3_max_silence_0=(
            0.3,
            5,
            0,
            1,
            {"aw": 0.3},
            [(3, 12), (15, 24), (36, 76)],
        ),
        mono_aw_0_3_max_silence_0_3=(0.3, 5, 0.3, 1, {"aw": 0.3}, [(3, 27), (36, 76)]),
        mono_aw_0_3_max_silence_0_5=(0.3, 5, 0.5, 1, {"aw": 0.3}, [(3, 27), (36, 76)]),
        mono_aw_0_3_max_silence_0_6=(0.3, 5, 0.6, 1, {"aw": 0.3}, [(3, 30), (36, 76)]),
        mono_aw_0_4_max_silence_0=(
            0.2,
            5,
            0,
            1,
            {"aw": 0.4},
            [(4, 12), (16, 24), (36, 76)],
        ),
        mono_aw_0_4_max_silence_0_3=(
            0.2,
            5,
            0.3,
            1,
            {"aw": 0.4},
            [(4, 12), (16, 24), (36, 76)],
        ),
        mono_aw_0_4_max_silence_0_4=(0.2, 5, 0.4, 1, {"aw": 0.4}, [(4, 28), (36, 76)]),
        stereo_uc_0_analysis_window_0_2=(
            0.2,
            5,
            0.2,
            2,
            {"uc": 0, "analysis_window": 0.2},
            [(2, 30), (34, 76)],
        ),
        stereo_uc_1_analysis_window_0_2=(
            0.2,
            5,
            0.2,
            2,
            {"uc": 1, "analysis_window": 0.2},
            [(10, 32), (36, 76)],
        ),
        stereo_uc_mix_aw_0_1_max_silence_0=(
            0.2,
            5,
            0,
            2,
            {"uc": "mix", "analysis_window": 0.1},
            [(10, 14), (17, 24), (26, 29), (36, 76)],
        ),
        stereo_uc_mix_aw_0_1_max_silence_0_1=(
            0.2,
            5,
            0.1,
            2,
            {"uc": "mix", "analysis_window": 0.1},
            [(10, 15), (17, 25), (26, 30), (36, 76)],
        ),
        stereo_uc_mix_aw_0_1_max_silence_0_2=(
            0.2,
            5,
            0.2,
            2,
            {"uc": "mix", "analysis_window": 0.1},
            [(10, 16), (17, 31), (36, 76)],
        ),
        stereo_uc_mix_aw_0_1_max_silence_0_3=(
            0.2,
            5,
            0.3,
            2,
            {"uc": "mix", "analysis_window": 0.1},
            [(10, 32), (36, 76)],
        ),
        stereo_uc_mix_aw_0_2_max_silence_0_min_dur_0_3=(
            0.3,
            5,
            0,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 14), (16, 24), (36, 76)],
        ),
        stereo_uc_mix_aw_0_2_max_silence_0_min_dur_0_41=(
            0.41,
            5,
            0,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(16, 24), (36, 76)],
        ),
        stereo_uc_mix_aw_0_2_max_silence_0_1=(
            0.2,
            5,
            0.1,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 14), (16, 24), (26, 28), (36, 76)],
        ),
        stereo_uc_mix_aw_0_2_max_silence_0_2=(
            0.2,
            5,
            0.2,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 30), (36, 76)],
        ),
        stereo_uc_mix_aw_0_2_max_silence_0_4=(
            0.2,
            5,
            0.4,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 32), (36, 76)],
        ),
        stereo_uc_mix_aw_0_2_max_silence_0_5=(
            0.2,
            5,
            0.5,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 32), (36, 76)],
        ),
        stereo_uc_mix_aw_0_2_max_silence_0_6=(
            0.2,
            5,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.2},
            [(10, 34), (36, 76)],
        ),
        stereo_uc_mix_aw_0_3_max_silence_0=(
            0.2,
            5,
            0,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 24), (27, 30), (36, 76)],
        ),
        stereo_uc_mix_aw_0_3_max_silence_0_min_dur_0_3=(
            0.4,
            5,
            0,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 24), (36, 76)],
        ),
        stereo_uc_mix_aw_0_3_max_silence_0_6=(
            0.2,
            5,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 57), (57, 76)],
        ),
        stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_1=(
            0.2,
            5.1,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 60), (60, 76)],
        ),
        stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_2=(
            0.2,
            5.2,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 60), (60, 76)],
        ),
        stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_3=(
            0.2,
            5.3,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 60), (60, 76)],
        ),
        stereo_uc_mix_aw_0_3_max_silence_0_6_max_dur_5_4=(
            0.2,
            5.4,
            0.6,
            2,
            {"uc": "mix", "analysis_window": 0.3},
            [(9, 63), (63, 76)],
        ),
        stereo_uc_mix_aw_0_4_max_silence_0=(
            0.2,
            5,
            0,
            2,
            {"uc": "mix", "analysis_window": 0.4},
            [(16, 24), (36, 76)],
        ),
        stereo_uc_mix_aw_0_4_max_silence_0_3=(
            0.2,
            5,
            0.3,
            2,
            {"uc": "mix", "analysis_window": 0.4},
            [(16, 24), (36, 76)],
        ),
        stereo_uc_mix_aw_0_4_max_silence_0_4=(
            0.2,
            5,
            0.4,
            2,
            {"uc": "mix", "analysis_window": 0.4},
            [(16, 28), (36, 76)],
        ),
    )
    def test_split_analysis_window(
        self, min_dur, max_dur, max_silence, channels, kwargs, expected
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
            **kwargs
        )

        region = AudioRegion(data, 10, 2, channels)
        regions_ar = region.split(
            min_dur=min_dur,
            max_dur=max_dur,
            max_silence=max_silence,
            drop_trailing_silence=False,
            strict_min_dur=False,
            **kwargs
        )

        regions = list(regions)
        regions_ar = list(regions_ar)
        err_msg = "Wrong number of regions after split, expected: "
        err_msg += "{}, found: {}".format(len(expected), len(regions))
        self.assertEqual(len(regions), len(expected), err_msg)
        err_msg = "Wrong number of regions after AudioRegion.split, expected: "
        err_msg += "{}, found: {}".format(len(expected), len(regions_ar))
        self.assertEqual(len(regions_ar), len(expected), err_msg)

        sample_width = 2
        sample_size_bytes = sample_width * channels
        for reg, reg_ar, exp in zip(regions, regions_ar, expected):
            onset, offset = exp
            exp_data = data[onset * sample_size_bytes : offset * sample_size_bytes]
            self.assertEqual(bytes(reg), exp_data)
            self.assertEqual(reg, reg_ar)

    @genty_dataset(
        filename_audio_format=(
            "tests/data/test_split_10HZ_stereo.raw",
            {"audio_format": "raw", "sr": 10, "sw": 2, "ch": 2},
        ),
        filename_audio_format_short_name=(
            "tests/data/test_split_10HZ_stereo.raw",
            {"fmt": "raw", "sr": 10, "sw": 2, "ch": 2},
        ),
        filename_no_audio_format=(
            "tests/data/test_split_10HZ_stereo.raw",
            {"sr": 10, "sw": 2, "ch": 2},
        ),
        filename_no_long_audio_params=(
            "tests/data/test_split_10HZ_stereo.raw",
            {"sampling_rate": 10, "sample_width": 2, "channels": 2},
        ),
        bytes_=(
            open("tests/data/test_split_10HZ_stereo.raw", "rb").read(),
            {"sr": 10, "sw": 2, "ch": 2},
        ),
        audio_reader=(
            AudioDataSource(
                "tests/data/test_split_10HZ_stereo.raw",
                sr=10,
                sw=2,
                ch=2,
                block_dur=0.1,
            ),
            {},
        ),
        audio_region=(
            AudioRegion(
                open("tests/data/test_split_10HZ_stereo.raw", "rb").read(), 10, 2, 2
            ),
            {},
        ),
        audio_source=(
            get_audio_source(
                "tests/data/test_split_10HZ_stereo.raw", sr=10, sw=2, ch=2
            ),
            {},
        ),
    )
    def test_split_input_type(self, input, kwargs):

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
        self.assertEqual(len(regions), len(expected), err_msg)
        for reg, exp in zip(regions, expected):
            onset, offset = exp
            exp_data = data[onset * sample_width * 2 : offset * sample_width * 2]
            self.assertEqual(bytes(reg), exp_data)

    @genty_dataset(
        min_dur_greater_than_max_dur=(0.5, 0.4, 0.1),
        durations_OK_but_wrong_number_of_analysis_windows=(0.44, 0.49, 0.1),
    )
    def test_split_wrong_min_max_dur(self, min_dur, max_dur, analysis_window):

        with self.assertRaises(ValueError) as val_err:
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
        self.assertEqual(err_msg, str(val_err.exception))

    @genty_dataset(
        max_silence_equals_max_dur=(0.5, 0.5, 0.1),
        max_silence_greater_than_max_dur=(0.5, 0.4, 0.1),
        durations_OK_but_wrong_number_of_analysis_windows=(0.44, 0.49, 0.1),
    )
    def test_split_wrong_max_silence_max_dur(
        self, max_silence, max_dur, analysis_window
    ):

        with self.assertRaises(ValueError) as val_err:
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
        self.assertEqual(err_msg, str(val_err.exception))

    @genty_dataset(
        negative_min_dur=({"min_dur": -1},),
        zero_min_dur=({"min_dur": 0},),
        negative_max_dur=({"max_dur": -1},),
        zero_max_dur=({"max_dur": 0},),
        negative_max_silence=({"max_silence": -1},),
        zero_analysis_window=({"analysis_window": 0},),
        negative_analysis_window=({"analysis_window": -1},),
    )
    def test_split_negative_temporal_params(self, wrong_param):

        params = {
            "min_dur": 0.2,
            "max_dur": 0.5,
            "max_silence": 0.1,
            "analysis_window": 0.1,
        }
        params.update(wrong_param)
        with self.assertRaises(ValueError) as val_err:
            split(None, **params)

        name = set(wrong_param).pop()
        value = wrong_param[name]
        err_msg = "'{}' ({}) must be >{} 0".format(
            name, value, "=" if name == "max_silence" else ""
        )
        self.assertEqual(err_msg, str(val_err.exception))

    def test_split_too_small_analysis_window(self):
        with self.assertRaises(ValueError) as val_err:
            split(b"", sr=10, sw=1, ch=1, analysis_window=0.09)
        err_msg = "Too small 'analysis_windows' (0.09) for sampling rate (10)."
        err_msg += " Analysis windows should at least be 1/10 to cover one "
        err_msg += "single data sample"
        self.assertEqual(err_msg, str(val_err.exception))
    
    def test_split_and_plot(self):

        with open("tests/data/test_split_10HZ_mono.raw", "rb") as fp:
            data = fp.read()

        region = AudioRegion(data, 10, 2, 1)
        with patch("auditok.core.plot_detections") as patch_fn:
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
        self.assertTrue(patch_fn.called)
        expected = [(2, 16), (17, 31), (34, 76)]
        sample_width = 2
        expected_regions = []
        for (onset, offset) in expected:
            onset *= sample_width
            offset *= sample_width
            expected_regions.append(AudioRegion(data[onset:offset], 10, 2, 1))
        self.assertEqual(regions, expected_regions)

@genty
class TestAudioRegion(TestCase):
    @genty_dataset(
        simple=(b"\0" * 8000, 0, 8000, 1, 1, 1, 1, 1000),
        one_ms_less_than_1_sec=(b"\0" * 7992, 0, 8000, 1, 1, 0.999, 0.999, 999),
        tree_quarter_ms_less_than_1_sec=(
            b"\0" * 7994,
            0,
            8000,
            1,
            1,
            0.99925,
            0.99925,
            999,
        ),
        half_ms_less_than_1_sec=(b"\0" * 7996, 0, 8000, 1, 1, 0.9995, 0.9995, 1000),
        quarter_ms_less_than_1_sec=(
            b"\0" * 7998,
            0,
            8000,
            1,
            1,
            0.99975,
            0.99975,
            1000,
        ),
        simple_sample_width_2=(b"\0" * 8000 * 2, 0, 8000, 2, 1, 1, 1, 1000),
        simple_stereo=(b"\0" * 8000 * 2, 0, 8000, 1, 2, 1, 1, 1000),
        simple_multichannel=(b"\0" * 8000 * 5, 0, 8000, 1, 5, 1, 1, 1000),
        simple_sample_width_2_multichannel=(
            b"\0" * 8000 * 2 * 5,
            0,
            8000,
            2,
            5,
            1,
            1,
            1000,
        ),
        one_ms_less_than_1s_sw_2_multichannel=(
            b"\0" * 7992 * 2 * 5,
            0,
            8000,
            2,
            5,
            0.999,
            0.999,
            999,
        ),
        tree_qrt_ms_lt_1_s_sw_2_multichannel=(
            b"\0" * 7994 * 2 * 5,
            0,
            8000,
            2,
            5,
            0.99925,
            0.99925,
            999,
        ),
        half_ms_lt_1s_sw_2_multichannel=(
            b"\0" * 7996 * 2 * 5,
            0,
            8000,
            2,
            5,
            0.9995,
            0.9995,
            1000,
        ),
        quarter_ms_lt_1s_sw_2_multichannel=(
            b"\0" * 7998 * 2 * 5,
            0,
            8000,
            2,
            5,
            0.99975,
            0.99975,
            1000,
        ),
        arbitrary_length_1=(
            b"\0" * int(8000 * 1.33),
            2.7,
            8000,
            1,
            1,
            4.03,
            1.33,
            1330,
        ),
        arbitrary_length_2=(
            b"\0" * int(8000 * 0.476),
            11.568,
            8000,
            1,
            1,
            12.044,
            0.476,
            476,
        ),
        arbitrary_length_sw_2_multichannel=(
            b"\0" * int(8000 * 1.711) * 2 * 3,
            9.415,
            8000,
            2,
            3,
            11.126,
            1.711,
            1711,
        ),
        arbitrary_samplig_rate=(
            b"\0" * int(3172 * 1.318),
            17.236,
            3172,
            1,
            1,
            17.236 + int(3172 * 1.318) / 3172,
            int(3172 * 1.318) / 3172,
            1318,
        ),
        arbitrary_sr_sw_2_multichannel=(
            b"\0" * int(11317 * 0.716) * 2 * 3,
            18.811,
            11317,
            2,
            3,
            18.811 + int(11317 * 0.716) / 11317,
            int(11317 * 0.716) / 11317,
            716,
        ),
    )
    def test_creation(
        self,
        data,
        start,
        sampling_rate,
        sample_width,
        channels,
        expected_end,
        expected_duration_s,
        expected_duration_ms,
    ):
        meta = {"start": start, "end": expected_end}
        region = AudioRegion(data, sampling_rate, sample_width, channels, meta)
        self.assertEqual(region.sampling_rate, sampling_rate)
        self.assertEqual(region.sr, sampling_rate)
        self.assertEqual(region.sample_width, sample_width)
        self.assertEqual(region.sw, sample_width)
        self.assertEqual(region.channels, channels)
        self.assertEqual(region.ch, channels)
        self.assertEqual(region.meta.start, start)
        self.assertEqual(region.meta.end, expected_end)
        self.assertEqual(region.duration, expected_duration_s)
        self.assertEqual(len(region.ms), expected_duration_ms)
        self.assertEqual(bytes(region), data)

    def test_creation_invalid_data_exception(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            _ = AudioRegion(
                data=b"ABCDEFGHI", sampling_rate=8, sample_width=2, channels=1
            )
        self.assertEqual(
            "The length of audio data must be an integer "
            "multiple of `sample_width * channels`",
            str(audio_param_err.exception),
        )

    @genty_dataset(
        simple=("output.wav", 1.230, "output.wav"),
        start=("output_{meta.start:g}.wav", 1.230, "output_1.23.wav"),
        start_2=("output_{meta.start}.wav", 1.233712, "output_1.233712.wav"),
        start_3=("output_{meta.start:.2f}.wav", 1.2300001, "output_1.23.wav"),
        start_4=("output_{meta.start:.3f}.wav", 1.233712, "output_1.234.wav"),
        start_5=("output_{meta.start:.8f}.wav", 1.233712, "output_1.23371200.wav"),
        start_end_duration=(
            "output_{meta.start}_{meta.end}_{duration}.wav",
            1.455,
            "output_1.455_2.455_1.0.wav",
        ),
        start_end_duration_2=(
            "output_{meta.start}_{meta.end}_{duration}.wav",
            1.455321,
            "output_1.455321_2.455321_1.0.wav",
        ),
    )
    def test_save(self, format, start, expected):
        with TemporaryDirectory() as tmpdir:
            region = AudioRegion(b"0" * 160, 160, 1, 1)
            meta = {"start": start, "end": start + region.duration}
            region.meta = meta
            format = os.path.join(tmpdir, format)
            filename = region.save(format)[len(tmpdir) + 1 :]
            self.assertEqual(filename, expected)

    def test_save_file_exists_exception(self):
        with TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "output.wav")
            open(filename, "w").close()
            region = AudioRegion(b"0" * 160, 160, 1, 1)
            with self.assertRaises(FileExistsError):
                region.save(filename, exists_ok=False)

    @genty_dataset(
        first_half=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(0, 500),
            b"a" * 80,
        ),
        second_half=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(500, None),
            b"b" * 80,
        ),
        second_half_negative=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-500, None),
            b"b" * 80,
        ),
        middle=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(200, 750),
            b"a" * 48 + b"b" * 40,
        ),
        middle_negative=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-800, -250),
            b"a" * 48 + b"b" * 40,
        ),
        middle_sw2=(
            AudioRegion(b"a" * 160 + b"b" * 160, 160, 2, 1),
            slice(200, 750),
            b"a" * 96 + b"b" * 80,
        ),
        middle_ch2=(
            AudioRegion(b"a" * 160 + b"b" * 160, 160, 1, 2),
            slice(200, 750),
            b"a" * 96 + b"b" * 80,
        ),
        middle_sw2_ch2=(
            AudioRegion(b"a" * 320 + b"b" * 320, 160, 2, 2),
            slice(200, 750),
            b"a" * 192 + b"b" * 160,
        ),
        but_first_sample=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(1, None),
            b"a" * (4000 - 8) + b"b" * 4000,
        ),
        but_first_sample_negative=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(-999, None),
            b"a" * (4000 - 8) + b"b" * 4000,
        ),
        but_last_sample=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(0, 999),
            b"a" * 4000 + b"b" * (4000 - 8),
        ),
        but_last_sample_negative=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(0, -1),
            b"a" * 4000 + b"b" * (4000 - 8),
        ),
        big_negative_start=(
            AudioRegion(b"a" * 160, 160, 1, 1),
            slice(-5000, None),
            b"a" * 160,
        ),
        big_negative_stop=(AudioRegion(b"a" * 160, 160, 1, 1), slice(None, -1500), b""),
        empty=(AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1), slice(0, 0), b""),
        empty_start_stop_reversed=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(200, 100),
            b"",
        ),
        empty_big_positive_start=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(2000, 3000),
            b"",
        ),
        empty_negative_reversed=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-100, -200),
            b"",
        ),
        empty_big_negative_stop=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(0, -2000),
            b"",
        ),
        arbitrary_sampling_rate=(
            AudioRegion(b"a" * 124 + b"b" * 376, 1234, 1, 1),
            slice(100, 200),
            b"a" + b"b" * 123,
        ),
    )
    def test_region_temporal_slicing(self, region, slice_, expected_data):
        sub_region = region.millis[slice_]
        self.assertEqual(bytes(sub_region), expected_data)
        start_sec = slice_.start / 1000 if slice_.start is not None else None
        stop_sec = slice_.stop / 1000 if slice_.stop is not None else None
        sub_region = region.sec[start_sec:stop_sec]
        self.assertEqual(bytes(sub_region), expected_data)

    @genty_dataset(
        first_half=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(0, 80),
            0,
            b"a" * 80,
        ),
        second_half=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(80, None),
            0.5,
            b"b" * 80,
        ),
        second_half_negative=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-80, None),
            0.5,
            b"b" * 80,
        ),
        middle=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(160 // 5, 160 // 4 * 3),
            0.2,
            b"a" * 48 + b"b" * 40,
        ),
        middle_negative=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-160 // 5 * 4, -160 // 4),
            0.2,
            b"a" * 48 + b"b" * 40,
        ),
        middle_sw2=(
            AudioRegion(b"a" * 160 + b"b" * 160, 160, 2, 1),
            slice(160 // 5, 160 // 4 * 3),
            0.2,
            b"a" * 96 + b"b" * 80,
        ),
        middle_ch2=(
            AudioRegion(b"a" * 160 + b"b" * 160, 160, 1, 2),
            slice(160 // 5, 160 // 4 * 3),
            0.2,
            b"a" * 96 + b"b" * 80,
        ),
        middle_sw2_ch2=(
            AudioRegion(b"a" * 320 + b"b" * 320, 160, 2, 2),
            slice(160 // 5, 160 // 4 * 3),
            0.2,
            b"a" * 192 + b"b" * 160,
        ),
        but_first_sample=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(1, None),
            1 / 8000,
            b"a" * (4000 - 1) + b"b" * 4000,
        ),
        but_first_sample_negative=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(-7999, None),
            1 / 8000,
            b"a" * (4000 - 1) + b"b" * 4000,
        ),
        but_last_sample=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(0, 7999),
            0,
            b"a" * 4000 + b"b" * (4000 - 1),
        ),
        but_last_sample_negative=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 8000, 1, 1),
            slice(0, -1),
            0,
            b"a" * 4000 + b"b" * (4000 - 1),
        ),
        big_negative_start=(
            AudioRegion(b"a" * 160, 160, 1, 1),
            slice(-1600, None),
            0,
            b"a" * 160,
        ),
        big_negative_stop=(
            AudioRegion(b"a" * 160, 160, 1, 1),
            slice(None, -1600),
            0,
            b"",
        ),
        empty=(AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1), slice(0, 0), 0, b""),
        empty_start_stop_reversed=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(80, 40),
            0.5,
            b"",
        ),
        empty_big_positive_start=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(1600, 3000),
            10,
            b"",
        ),
        empty_negative_reversed=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(-16, -32),
            0.9,
            b"",
        ),
        empty_big_negative_stop=(
            AudioRegion(b"a" * 80 + b"b" * 80, 160, 1, 1),
            slice(0, -2000),
            0,
            b"",
        ),
        arbitrary_sampling_rate=(
            AudioRegion(b"a" * 124 + b"b" * 376, 1235, 1, 1),
            slice(100, 200),
            100 / 1235,
            b"a" * 24 + b"b" * 76,
        ),
        arbitrary_sampling_rate_middle_sw2_ch2=(
            AudioRegion(b"a" * 124 + b"b" * 376, 1235, 2, 2),
            slice(25, 50),
            25 / 1235,
            b"a" * 24 + b"b" * 76,
        ),
    )
    def test_region_sample_slicing(self, region, slice_, time_shift, expected_data):
        sub_region = region[slice_]
        self.assertEqual(bytes(sub_region), expected_data)

    @genty_dataset(
        simple=(8000, 1, 1),
        stereo_sw_2=(8000, 2, 2),
        arbitrary_sr_multichannel=(5413, 2, 3),
    )
    def test_concatenation(self, sampling_rate, sample_width, channels):

        region_1, region_2 = _make_random_length_regions(
            [b"a", b"b"], sampling_rate, sample_width, channels
        )
        expected_duration = region_1.duration + region_2.duration
        expected_data = bytes(region_1) + bytes(region_2)
        concat_region = region_1 + region_2
        self.assertAlmostEqual(concat_region.duration, expected_duration, places=6)
        self.assertEqual(bytes(concat_region), expected_data)

    @genty_dataset(
        simple=(8000, 1, 1),
        stereo_sw_2=(8000, 2, 2),
        arbitrary_sr_multichannel=(5413, 2, 3),
    )
    def test_concatenation_many(self, sampling_rate, sample_width, channels):

        regions = _make_random_length_regions(
            [b"a", b"b", b"c"], sampling_rate, sample_width, channels
        )
        expected_duration = sum(r.duration for r in regions)
        expected_data = b"".join(bytes(r) for r in regions)
        concat_region = sum(regions)

        self.assertAlmostEqual(concat_region.duration, expected_duration, places=6)
        self.assertEqual(bytes(concat_region), expected_data)

    def test_concatenation_different_sampling_rate_error(self):

        region_1 = AudioRegion(b"a" * 100, 8000, 1, 1)
        region_2 = AudioRegion(b"b" * 100, 3000, 1, 1)

        with self.assertRaises(ValueError) as val_err:
            region_1 + region_2
        self.assertEqual(
            "Can only concatenate AudioRegions of the same "
            "sampling rate (8000 != 3000)",
            str(val_err.exception),
        )

    def test_concatenation_different_sample_width_error(self):

        region_1 = AudioRegion(b"a" * 100, 8000, 2, 1)
        region_2 = AudioRegion(b"b" * 100, 8000, 4, 1)

        with self.assertRaises(ValueError) as val_err:
            region_1 + region_2
        self.assertEqual(
            "Can only concatenate AudioRegions of the same " "sample width (2 != 4)",
            str(val_err.exception),
        )

    def test_concatenation_different_number_of_channels_error(self):

        region_1 = AudioRegion(b"a" * 100, 8000, 1, 1)
        region_2 = AudioRegion(b"b" * 100, 8000, 1, 2)

        with self.assertRaises(ValueError) as val_err:
            region_1 + region_2
        self.assertEqual(
            "Can only concatenate AudioRegions of the same "
            "number of channels (1 != 2)",
            str(val_err.exception),
        )

    @genty_dataset(
        simple=(0.01, 0.03, 240, 30),
        rounded_len_floor=(0.00575, 0.01725, 138, 17),
        rounded_len_ceil=(0.00625, 0.01875, 150, 19),
    )
    def test_multiplication(
        self, duration, expected_duration, expected_len, expected_len_ms
    ):
        sw = 2
        data = b"0" * int(duration * 8000 * sw)
        region = AudioRegion(data, 8000, sw, 1)
        m_region = 1 * region * 3
        self.assertEqual(bytes(m_region), data * 3)
        self.assertEqual(m_region.sr, 8000)
        self.assertEqual(m_region.sw, 2)
        self.assertEqual(m_region.ch, 1)
        self.assertEqual(m_region.duration, expected_duration)
        self.assertEqual(len(m_region), expected_len)
        self.assertEqual(m_region.len, expected_len)
        self.assertEqual(m_region.s.len, expected_duration)
        self.assertEqual(len(m_region.ms), expected_len_ms)
        self.assertEqual(m_region.ms.len, expected_len_ms)

    @genty_dataset(_str=("x", "str"), _float=(1.4, "float"))
    def test_multiplication_non_int(self, factor, _type):
        with self.assertRaises(TypeError) as type_err:
            AudioRegion(b"0" * 80, 8000, 1, 1) * factor
            err_msg = "Can't multiply AudioRegion by a non-int of type '{}'"
            self.assertEqual(err_msg.format(_type), str(type_err.exception))

    @genty_dataset(
        simple=([b"a" * 80, b"b" * 80],),
        extra_samples_1=([b"a" * 31, b"b" * 31, b"c" * 30],),
        extra_samples_2=([b"a" * 31, b"b" * 30, b"c" * 30],),
        extra_samples_3=([b"a" * 11, b"b" * 11, b"c" * 10, b"c" * 10],),
    )
    def test_truediv(self, data):

        region = AudioRegion(b"".join(data), 80, 1, 1)

        sub_regions = region / len(data)
        for data_i, region in zip(data, sub_regions):
            self.assertEqual(len(data_i), len(bytes(region)))

    @genty_dataset(
        mono_sw_1=(b"a" * 10, 1, 1, "b", [97] * 10),
        mono_sw_2=(b"a" * 10, 2, 1, "h", [24929] * 5),
        mono_sw_4=(b"a" * 8, 4, 1, "i", [1633771873] * 2),
        stereo_sw_1=(b"ab" * 5, 1, 2, "b", [[97] * 5, [98] * 5]),
    )
    def test_samples(self, data, sample_width, channels, fmt, expected):

        region = AudioRegion(data, 10, sample_width, channels)
        if isinstance(expected[0], list):
            expected = [array_(fmt, exp) for exp in expected]
        else:
            expected = array_(fmt, expected)
        samples = region.samples
        equal = samples == expected
        try:
            # for numpy
            equal = equal.all()
        except:
            pass
        self.assertTrue(equal)
