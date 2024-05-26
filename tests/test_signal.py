from array import array as array_

import numpy as np
import pytest

from auditok import signal as signal_
from auditok import signal_numpy


@pytest.fixture
def setup_data():
    return b"012345679ABC"


@pytest.fixture
def numpy_fmt():
    return {"b": np.int8, "h": np.int16, "i": np.int32}


@pytest.mark.parametrize(
    "sample_width, expected",
    [
        (1, [[48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67]]),  # int8_mono
        (2, [[12592, 13106, 13620, 14134, 16697, 17218]]),  # int16_mono
        (4, [[858927408, 926299444, 1128415545]]),  # int32_mono
        (
            1,
            [[48, 50, 52, 54, 57, 66], [49, 51, 53, 55, 65, 67]],
        ),  # int8_stereo
        (2, [[12592, 13620, 16697], [13106, 14134, 17218]]),  # int16_stereo
        (4, [[858927408], [926299444], [1128415545]]),  # int32_3channel
    ],
    ids=[
        "int8_mono",
        "int16_mono",
        "int32_mono",
        "int8_stereo",
        "int16_stereo",
        "int32_3channel",
    ],
)
def test_to_array(setup_data, sample_width, expected):
    data = setup_data
    channels = len(expected)
    expected = [array_(signal_.FORMAT[sample_width], xi) for xi in expected]
    result = signal_.to_array(data, sample_width, channels)
    result_numpy = signal_numpy.to_array(data, sample_width, channels)
    assert result == expected
    assert (result_numpy == np.asarray(expected)).all()
    assert result_numpy.dtype == np.float64


@pytest.mark.parametrize(
    "fmt, channels, selected, expected",
    [
        (
            "b",
            1,
            0,
            [48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67],
        ),  # int8_1channel_select_0
        ("b", 2, 0, [48, 50, 52, 54, 57, 66]),  # int8_2channel_select_0
        ("b", 3, 0, [48, 51, 54, 65]),  # int8_3channel_select_0
        ("b", 3, 1, [49, 52, 55, 66]),  # int8_3channel_select_1
        ("b", 3, 2, [50, 53, 57, 67]),  # int8_3channel_select_2
        ("b", 4, 0, [48, 52, 57]),  # int8_4channel_select_0
        (
            "h",
            1,
            0,
            [12592, 13106, 13620, 14134, 16697, 17218],
        ),  # int16_1channel_select_0
        ("h", 2, 0, [12592, 13620, 16697]),  # int16_2channel_select_0
        ("h", 2, 1, [13106, 14134, 17218]),  # int16_2channel_select_1
        ("h", 3, 0, [12592, 14134]),  # int16_3channel_select_0
        ("h", 3, 1, [13106, 16697]),  # int16_3channel_select_1
        ("h", 3, 2, [13620, 17218]),  # int16_3channel_select_2
        (
            "i",
            1,
            0,
            [858927408, 926299444, 1128415545],
        ),  # int32_1channel_select_0
        ("i", 3, 0, [858927408]),  # int32_3channel_select_0
        ("i", 3, 1, [926299444]),  # int32_3channel_select_1
        ("i", 3, 2, [1128415545]),  # int32_3channel_select_2
    ],
    ids=[
        "int8_1channel_select_0",
        "int8_2channel_select_0",
        "int8_3channel_select_0",
        "int8_3channel_select_1",
        "int8_3channel_select_2",
        "int8_4channel_select_0",
        "int16_1channel_select_0",
        "int16_2channel_select_0",
        "int16_2channel_select_1",
        "int16_3channel_select_0",
        "int16_3channel_select_1",
        "int16_3channel_select_2",
        "int32_1channel_select_0",
        "int32_3channel_select_0",
        "int32_3channel_select_1",
        "int32_3channel_select_2",
    ],
)
def test_extract_single_channel(
    setup_data, numpy_fmt, fmt, channels, selected, expected
):
    data = setup_data
    result = signal_.extract_single_channel(data, fmt, channels, selected)
    expected = array_(fmt, expected)
    expected_numpy_fmt = numpy_fmt[fmt]
    assert result == expected
    result_numpy = signal_numpy.extract_single_channel(
        data, numpy_fmt[fmt], channels, selected
    )
    assert all(result_numpy == expected)
    assert result_numpy.dtype == expected_numpy_fmt


@pytest.mark.parametrize(
    "fmt, channels, expected",
    [
        ("b", 2, [48, 50, 52, 54, 61, 66]),  # int8_2channel
        ("b", 4, [50, 54, 64]),  # int8_4channel
        ("h", 1, [12592, 13106, 13620, 14134, 16697, 17218]),  # int16_1channel
        ("h", 2, [12849, 13877, 16958]),  # int16_2channel
        ("i", 3, [971214132]),  # int32_3channel
    ],
    ids=[
        "int8_2channel",
        "int8_4channel",
        "int16_1channel",
        "int16_2channel",
        "int32_3channel",
    ],
)
def test_compute_average_channel(
    setup_data, numpy_fmt, fmt, channels, expected
):
    data = setup_data
    result = signal_.compute_average_channel(data, fmt, channels)
    expected = array_(fmt, expected)
    expected_numpy_fmt = numpy_fmt[fmt]
    assert result == expected
    result_numpy = signal_numpy.compute_average_channel(
        data, numpy_fmt[fmt], channels
    )
    assert all(result_numpy == expected)
    assert result_numpy.dtype == expected_numpy_fmt


@pytest.mark.parametrize(
    "sample_width, expected",
    [
        (1, [48, 50, 52, 54, 61, 66]),  # int8_2channel
        (2, [12849, 13877, 16957]),  # int16_2channel
    ],
    ids=["int8_2channel", "int16_2channel"],
)
def test_compute_average_channel_stereo(setup_data, sample_width, expected):
    data = setup_data
    result = signal_.compute_average_channel_stereo(data, sample_width)
    fmt = signal_.FORMAT[sample_width]
    expected = array_(fmt, expected)
    assert result == expected


@pytest.mark.parametrize(
    "fmt, channels, expected",
    [
        (
            "b",
            1,
            [[48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67]],
        ),  # int8_1channel
        (
            "b",
            2,
            [[48, 50, 52, 54, 57, 66], [49, 51, 53, 55, 65, 67]],
        ),  # int8_2channel
        (
            "b",
            4,
            [[48, 52, 57], [49, 53, 65], [50, 54, 66], [51, 55, 67]],
        ),  # int8_4channel
        (
            "h",
            2,
            [[12592, 13620, 16697], [13106, 14134, 17218]],
        ),  # int16_2channel
        ("i", 3, [[858927408], [926299444], [1128415545]]),  # int32_3channel
    ],
    ids=[
        "int8_1channel",
        "int8_2channel",
        "int8_4channel",
        "int16_2channel",
        "int32_3channel",
    ],
)
def test_separate_channels(setup_data, numpy_fmt, fmt, channels, expected):
    data = setup_data
    result = signal_.separate_channels(data, fmt, channels)
    expected = [array_(fmt, exp) for exp in expected]
    expected_numpy_fmt = numpy_fmt[fmt]
    assert result == expected
    result_numpy = signal_numpy.separate_channels(
        data, numpy_fmt[fmt], channels
    )
    assert (result_numpy == expected).all()
    assert result_numpy.dtype == expected_numpy_fmt


@pytest.mark.parametrize(
    "x, sample_width, expected",
    [
        ([300, 320, 400, 600], 2, 52.50624901923348),  # simple
        ([0], 2, -200),  # zero
        ([0, 0, 0], 2, -200),  # zeros
    ],
    ids=["simple", "zero", "zeros"],
)
def test_calculate_energy_single_channel(x, sample_width, expected):
    x = array_(signal_.FORMAT[sample_width], x)
    energy = signal_.calculate_energy_single_channel(x, sample_width)
    assert energy == expected
    energy = signal_numpy.calculate_energy_single_channel(x, sample_width)
    assert energy == expected


@pytest.mark.parametrize(
    "x, sample_width, aggregation_fn, expected",
    [
        (
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            2,
            min,
            46.485649105953854,
        ),  # min_
        (
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            2,
            max,
            52.50624901923348,
        ),  # max_
    ],
    ids=["min_", "max_"],
)
def test_calculate_energy_multichannel(
    x, sample_width, aggregation_fn, expected
):
    x = [array_(signal_.FORMAT[sample_width], xi) for xi in x]
    energy = signal_.calculate_energy_multichannel(
        x, sample_width, aggregation_fn
    )
    assert energy == expected
    energy = signal_numpy.calculate_energy_multichannel(
        x, sample_width, aggregation_fn
    )
    assert energy == expected
