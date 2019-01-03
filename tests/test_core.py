import unittest
from genty import genty, genty_dataset
from auditok import AudioRegion


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
