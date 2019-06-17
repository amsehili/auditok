import os
from unittest import TestCase
from random import random
from tempfile import TemporaryDirectory
from genty import genty, genty_dataset
from auditok import split, AudioRegion, AudioParameterError
from auditok.io import (
    _normalize_use_channel,
    _extract_selected_channel,
    get_audio_source,
)


def _make_random_length_regions(
    byte_seq, sampling_rate, sample_width, channels
):
    regions = []
    for b in byte_seq:
        duration = round(random() * 10, 6)
        data = b * int(duration * sampling_rate) * sample_width * channels
        start = round(random() * 13, 3)
        region = AudioRegion(
            data, start, sampling_rate, sample_width, channels
        )
        regions.append(region)
    return regions


@genty
class TestSplit(TestCase):
    @genty_dataset(
        simple=(
            0.2,
            5,
            0.2,
            False,
            False,
            {"eth": 50},
            [(2, 16), (17, 31), (34, 76)],
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
        high_energy_threshold=(
            0.2,
            5,
            0.2,
            False,
            False,
            {"energy_threshold": 60},
            [],
        ),
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
        drop_trailing_silence_2=(
            1.5,
            5,
            0.2,
            True,
            False,
            {"eth": 50},
            [(34, 76)],
        ),
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
        regions = list(regions)
        err_msg = "Wrong number of regions after split, expected: "
        err_msg += "{}, found: {}".format(len(expected), len(regions))
        self.assertEqual(len(regions), len(expected), err_msg)

        sample_width = 2
        for reg, exp in zip(regions, expected):
            onset, offset = exp
            exp_data = data[onset * sample_width : offset * sample_width]
            self.assertEqual(bytes(reg), exp_data)


@genty
class TestAudioRegion(TestCase):
    @genty_dataset(
        simple=(b"\0" * 8000, 0, 8000, 1, 1, 1, 1, 1000),
        one_ms_less_than_1_sec=(
            b"\0" * 7992,
            0,
            8000,
            1,
            1,
            0.999,
            0.999,
            999,
        ),
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
        half_ms_less_than_1_sec=(
            b"\0" * 7996,
            0,
            8000,
            1,
            1,
            0.9995,
            0.9995,
            1000,
        ),
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
        region = AudioRegion(
            data, start, sampling_rate, sample_width, channels
        )
        self.assertEqual(region.sampling_rate, sampling_rate)
        self.assertEqual(region.sr, sampling_rate)
        self.assertEqual(region.sample_width, sample_width)
        self.assertEqual(region.sw, sample_width)
        self.assertEqual(region.channels, channels)
        self.assertEqual(region.ch, channels)
        self.assertEqual(region.start, start)
        self.assertEqual(region.end, expected_end)
        self.assertEqual(region.duration, expected_duration_s)
        self.assertEqual(len(region), expected_duration_ms)
        self.assertEqual(bytes(region), data)

    def test_creation_invalid_data_exception(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            _ = AudioRegion(
                data=b"ABCDEFGHI",
                start=0,
                sampling_rate=8,
                sample_width=2,
                channels=1,
            )
        self.assertEqual(
            "The length of audio data must be an integer "
            "multiple of `sample_width * channels`",
            str(audio_param_err.exception),
        )

    @genty_dataset(
        simple=("output.wav", 1.230, "output.wav"),
        start=("output_{start}.wav", 1.230, "output_1.23.wav"),
        start_2=("output_{start}.wav", 1.233712, "output_1.233712.wav"),
        start_3=("output_{start}.wav", 1.2300001, "output_1.23.wav"),
        start_4=("output_{start:.3f}.wav", 1.233712, "output_1.234.wav"),
        start_5=(
            "output_{start:.8f}.wav",
            1.233712345,
            "output_1.23371200.wav",
        ),
        start_end_duration=(
            "output_{start}_{end}_{duration}.wav",
            1.455,
            "output_1.455_2.455_1.0.wav",
        ),
        start_end_duration_2=(
            "output_{start}_{end}_{duration}.wav",
            1.455321,
            "output_1.455321_2.455321_1.0.wav",
        ),
    )
    def test_save(self, format, start, expected):
        with TemporaryDirectory() as tmpdir:
            region = AudioRegion(b"0" * 160, start, 160, 1, 1)
            format = os.path.join(tmpdir, format)
            filename = region.save(format)[len(tmpdir) + 1 :]
            self.assertEqual(filename, expected)

    def test_save_file_exists_exception(self):
        with TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "output.wav")
            open(filename, "w").close()
            region = AudioRegion(b"0" * 160, 0, 160, 1, 1)
            with self.assertRaises(FileExistsError):
                region.save(filename, exists_ok=False)

    @genty_dataset(
        first_half=(
            AudioRegion(b"a" * 80 + b"b" * 80, 0, 160, 1, 1),
            slice(0, 500),
            0,
            b"a" * 80,
        ),
        second_half=(
            AudioRegion(b"a" * 80 + b"b" * 80, 0, 160, 1, 1),
            slice(500, None),
            0.5,
            b"b" * 80,
        ),
        second_half_negative=(
            AudioRegion(b"a" * 80 + b"b" * 80, 0, 160, 1, 1),
            slice(-500, None),
            0.5,
            b"b" * 80,
        ),
        middle=(
            AudioRegion(b"a" * 80 + b"b" * 80, 0, 160, 1, 1),
            slice(200, 750),
            0.2,
            b"a" * 48 + b"b" * 40,
        ),
        middle_negative=(
            AudioRegion(b"a" * 80 + b"b" * 80, 0, 160, 1, 1),
            slice(-800, -250),
            0.2,
            b"a" * 48 + b"b" * 40,
        ),
        middle_sw2=(
            AudioRegion(b"a" * 160 + b"b" * 160, 0, 160, 2, 1),
            slice(200, 750),
            0.2,
            b"a" * 96 + b"b" * 80,
        ),
        middle_ch2=(
            AudioRegion(b"a" * 160 + b"b" * 160, 0, 160, 1, 2),
            slice(200, 750),
            0.2,
            b"a" * 96 + b"b" * 80,
        ),
        middle_sw2_ch2=(
            AudioRegion(b"a" * 320 + b"b" * 320, 0, 160, 2, 2),
            slice(200, 750),
            0.2,
            b"a" * 192 + b"b" * 160,
        ),
        but_first_sample=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 0, 8000, 1, 1),
            slice(1, None),
            0.001,
            b"a" * (4000 - 8) + b"b" * 4000,
        ),
        but_first_sample_negative=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 0, 8000, 1, 1),
            slice(-999, None),
            0.001,
            b"a" * (4000 - 8) + b"b" * 4000,
        ),
        but_last_sample=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 0, 8000, 1, 1),
            slice(0, 999),
            0,
            b"a" * 4000 + b"b" * (4000 - 8),
        ),
        but_last_sample_negative=(
            AudioRegion(b"a" * 4000 + b"b" * 4000, 0, 8000, 1, 1),
            slice(0, -1),
            0,
            b"a" * 4000 + b"b" * (4000 - 8),
        ),
        big_negative_start=(
            AudioRegion(b"a" * 160, 0, 160, 1, 1),
            slice(-5000, None),
            0,
            b"a" * 160,
        ),
        big_negative_stop=(
            AudioRegion(b"a" * 160, 0, 160, 1, 1),
            slice(None, -1500),
            0,
            b"",
        ),
        empty=(
            AudioRegion(b"a" * 80 + b"b" * 80, 0, 160, 1, 1),
            slice(0, 0),
            0,
            b"",
        ),
        empty_start_stop_reversed=(
            AudioRegion(b"a" * 80 + b"b" * 80, 0, 160, 1, 1),
            slice(200, 100),
            0.2,
            b"",
        ),
        empty_big_positive_start=(
            AudioRegion(b"a" * 80 + b"b" * 80, 0, 160, 1, 1),
            slice(2000, 3000),
            2,
            b"",
        ),
        empty_negative_reversed=(
            AudioRegion(b"a" * 80 + b"b" * 80, 0, 160, 1, 1),
            slice(-100, -200),
            0.9,
            b"",
        ),
        empty_big_negative_stop=(
            AudioRegion(b"a" * 80 + b"b" * 80, 0, 160, 1, 1),
            slice(0, -2000),
            0,
            b"",
        ),
    )
    def test_region_slicing(
        self, region, slice_, expected_start, expected_data
    ):
        sub_region = region[slice_]
        self.assertEqual(sub_region.start, expected_start)
        self.assertEqual(bytes(sub_region), expected_data)

    @genty_dataset(
        simple=(8000, 1, 1),
        stereo_sw_2=(8000, 2, 2),
        arbitray_sr_multichannel=(5413, 2, 3),
    )
    def test_concatenation(self, sampling_rate, sample_width, channels):

        region_1, region_2 = _make_random_length_regions(
            [b"a", b"b"], sampling_rate, sample_width, channels
        )

        expected_start = region_1.start
        expected_duration = region_1.duration + region_2.duration
        expected_end = expected_start + expected_duration
        expected_data = bytes(region_1) + bytes(region_2)
        concat_region = region_1 + region_2

        self.assertEqual(concat_region.start, expected_start)
        self.assertAlmostEqual(concat_region.end, expected_end, places=6)
        self.assertAlmostEqual(
            concat_region.duration, expected_duration, places=6
        )
        self.assertEqual(bytes(concat_region), expected_data)

    @genty_dataset(
        simple=(8000, 1, 1),
        stereo_sw_2=(8000, 2, 2),
        arbitray_sr_multichannel=(5413, 2, 3),
    )
    def test_concatenation_many(self, sampling_rate, sample_width, channels):

        regions = _make_random_length_regions(
            [b"a", b"b", b"c"], sampling_rate, sample_width, channels
        )
        expected_start = regions[0].start
        expected_duration = sum(r.duration for r in regions)
        expected_end = expected_start + expected_duration
        expected_data = b"".join(bytes(r) for r in regions)
        concat_region = sum(regions)

        self.assertEqual(concat_region.start, expected_start)
        self.assertAlmostEqual(concat_region.end, expected_end, places=6)
        self.assertAlmostEqual(
            concat_region.duration, expected_duration, places=6
        )
        self.assertEqual(bytes(concat_region), expected_data)

    def test_concatenation_different_sampling_rate_error(self):

        region_1 = AudioRegion(b"a" * 100, 0, 8000, 1, 1)
        region_2 = AudioRegion(b"b" * 100, 0, 3000, 1, 1)

        with self.assertRaises(ValueError) as val_err:
            region_1 + region_2
        self.assertEqual(
            "Can only concatenate AudioRegions of the same "
            "sampling rate (8000 != 3000)",
            str(val_err.exception),
        )

    def test_concatenation_different_sample_width_error(self):

        region_1 = AudioRegion(b"a" * 100, 0, 8000, 2, 1)
        region_2 = AudioRegion(b"b" * 100, 0, 8000, 4, 1)

        with self.assertRaises(ValueError) as val_err:
            region_1 + region_2
        self.assertEqual(
            "Can only concatenate AudioRegions of the same "
            "sample width (2 != 4)",
            str(val_err.exception),
        )

    def test_concatenation_different_number_of_channels_error(self):

        region_1 = AudioRegion(b"a" * 100, 0, 8000, 1, 1)
        region_2 = AudioRegion(b"b" * 100, 0, 8000, 1, 2)

        with self.assertRaises(ValueError) as val_err:
            region_1 + region_2
        self.assertEqual(
            "Can only concatenate AudioRegions of the same "
            "number of channels (1 != 2)",
            str(val_err.exception),
        )

    @genty_dataset(
        simple=(0.01, 0.03, 30),
        rounded_len_floor=(0.00575, 0.01725, 17),
        rounded_len_ceil=(0.00625, 0.01875, 19),
    )
    def test_multiplication(
        self, duration, expected_duration, expected_length
    ):
        sw = 2
        data = b"0" * int(duration * 8000 * sw)
        region = AudioRegion(data, 0, 8000, sw, 1)
        m_region = 1 * region * 3
        self.assertEqual(bytes(m_region), data * 3)
        self.assertEqual(m_region.sr, 8000)
        self.assertEqual(m_region.sw, 2)
        self.assertEqual(m_region.ch, 1)
        self.assertEqual(m_region.duration, expected_duration)
        self.assertEqual(len(m_region), expected_length)

    @genty_dataset(_str=("x", "str"), _float=(1.4, "float"))
    def test_multiplication_non_int(self, factor, _type):
        with self.assertRaises(TypeError) as type_err:
            AudioRegion(b"0" * 80, 0, 8000, 1, 1) * factor
            err_msg = "Can't multiply AudioRegion by a non-int of type '{}'"
            self.assertEqual(err_msg.format(_type), str(type_err.exception))
