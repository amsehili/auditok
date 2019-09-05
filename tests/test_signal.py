from unittest import TestCase
from array import array as array_
from genty import genty, genty_dataset
import numpy as np
from auditok import signal as signal_
from auditok import signal_numpy


@genty
class TestSignal(TestCase):
    def setUp(self):
        self.data = b"012345679ABC"
        self.numpy_fmt = {"b": np.int8, "h": np.int16, "i": np.int32}

    @genty_dataset(
        int8=("b", [48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67]),
        int16=("h", [12592, 13106, 13620, 14134, 16697, 17218]),
        int32=("i", [858927408, 926299444, 1128415545]),
    )
    def test_to_array(self, fmt, expected):
        resutl = signal_.to_array(self.data, fmt)
        expected = array_(fmt, expected)
        self.assertEqual(resutl, expected)
        resutl_numpy = signal_numpy.to_array(self.data, self.numpy_fmt[fmt])
        self.assertTrue(all(resutl_numpy == expected))
        self.assertEqual(resutl_numpy.dtype, np.float64)

    @genty_dataset(
        int8_1channel_select_0=(
            "b",
            1,
            0,
            [48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67],
        ),
        int8_2channel_select_0=("b", 2, 0, [48, 50, 52, 54, 57, 66]),
        int8_3channel_select_0=("b", 3, 0, [48, 51, 54, 65]),
        int8_3channel_select_1=("b", 3, 1, [49, 52, 55, 66]),
        int8_3channel_select_2=("b", 3, 2, [50, 53, 57, 67]),
        int8_4channel_select_0=("b", 4, 0, [48, 52, 57]),
        int16_1channel_select_0=(
            "h",
            1,
            0,
            [12592, 13106, 13620, 14134, 16697, 17218],
        ),
        int16_2channel_select_0=("h", 2, 0, [12592, 13620, 16697]),
        int16_2channel_select_1=("h", 2, 1, [13106, 14134, 17218]),
        int16_3channel_select_0=("h", 3, 0, [12592, 14134]),
        int16_3channel_select_1=("h", 3, 1, [13106, 16697]),
        int16_3channel_select_2=("h", 3, 2, [13620, 17218]),
        int32_1channel_select_0=(
            "i",
            1,
            0,
            [858927408, 926299444, 1128415545],
        ),
        int32_3channel_select_0=("i", 3, 0, [858927408]),
        int32_3channel_select_1=("i", 3, 1, [926299444]),
        int32_3channel_select_2=("i", 3, 2, [1128415545]),
    )
    def test_extract_single_channel(self, fmt, channels, selected, expected):
        resutl = signal_.extract_single_channel(
            self.data, fmt, channels, selected
        )
        expected = array_(fmt, expected)
        self.assertEqual(resutl, expected)
        resutl_numpy = signal_numpy.extract_single_channel(
            self.data, self.numpy_fmt[fmt], channels, selected
        )
        self.assertTrue(all(resutl_numpy == expected))
        self.assertEqual(resutl_numpy.dtype, np.float64)

    @genty_dataset(
        int8_2channel=("b", 2, [48, 50, 52, 54, 61, 66]),
        int8_4channel=("b", 4, [50, 54, 64]),
        int16_1channel=("h", 1, [12592, 13106, 13620, 14134, 16697, 17218]),
        int16_2channel=("h", 2, [12849, 13877, 16958]),
        int32_3channel=("i", 3, [971214132]),
    )
    def test_average_channels(self, fmt, channels, expected):
        resutl = signal_.average_channels(self.data, fmt, channels)
        expected = array_(fmt, expected)
        self.assertEqual(resutl, expected)
        resutl_numpy = signal_numpy.average_channels(
            self.data, self.numpy_fmt[fmt], channels
        )
        self.assertTrue(all(resutl_numpy == expected))
        self.assertEqual(resutl_numpy.dtype, np.float64)

    @genty_dataset(
        int8_1channel=(
            "b",
            1,
            [[48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67]],
        ),
        int8_2channel=(
            "b",
            2,
            [[48, 50, 52, 54, 57, 66], [49, 51, 53, 55, 65, 67]],
        ),
        int8_4channel=(
            "b",
            4,
            [[48, 52, 57], [49, 53, 65], [50, 54, 66], [51, 55, 67]],
        ),
        int16_2channel=(
            "h",
            2,
            [[12592, 13620, 16697], [13106, 14134, 17218]],
        ),
        int32_3channel=("i", 3, [[858927408], [926299444], [1128415545]]),
    )
    def test_separate_channels(self, fmt, channels, expected):
        resutl = signal_.separate_channels(self.data, fmt, channels)
        expected = [array_(fmt, exp) for exp in expected]
        self.assertEqual(resutl, expected)

        resutl_numpy = signal_numpy.separate_channels(
            self.data, self.numpy_fmt[fmt], channels
        )
        self.assertTrue((resutl_numpy == expected).all())
        self.assertEqual(resutl_numpy.dtype, np.float64)

    @genty_dataset(
        simple=([300, 320, 400, 600], 52.506639194632434),
        zero=([0], -200),
        zeros=([0, 0, 0], -200),
    )
    def test_calculate_energy_single_channel(self, x, expected):
        energy = signal_.calculate_energy_single_channel(x)
        self.assertEqual(energy, expected)
        energy = signal_numpy.calculate_energy_single_channel(x)
        self.assertEqual(energy, expected)

    @genty_dataset(
        min_=(
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            min,
            46.48603928135281,
        ),
        max_=(
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            max,
            52.506639194632434,
        ),
    )
    def test_calculate_energy_multichannel(self, x, aggregation_fn, expected):
        energy = signal_.calculate_energy_multichannel(x, aggregation_fn)
        self.assertEqual(energy, expected)

        energy = signal_numpy.calculate_energy_multichannel(x, aggregation_fn)
        self.assertEqual(energy, expected)
