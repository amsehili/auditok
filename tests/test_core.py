import os
import unittest
from random import random
from tempfile import TemporaryDirectory
from genty import genty, genty_dataset
from auditok import AudioRegion, AudioParameterError


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
class TestAudioRegion(unittest.TestCase):
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
        # see test_concatenation
        self.assertEqual(len(concat_region), round(expected_duration * 1000))

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
