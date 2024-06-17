from array import array as array_

import numpy as np
import pytest

from auditok import signal

# from auditok import signal as signal_
# from auditok import signal


@pytest.fixture
def setup_data():
    return b"012345679ABC"


@pytest.mark.parametrize(
    "sample_width, expected",
    [
        (
            1,
            [[48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67]],
        ),  # int8_1channel
        (
            1,
            [[48, 50, 52, 54, 57, 66], [49, 51, 53, 55, 65, 67]],
        ),  # int8_2channel
        (
            1,
            [[48, 52, 57], [49, 53, 65], [50, 54, 66], [51, 55, 67]],
        ),  # int8_4channel
        (2, [[12592, 13106, 13620, 14134, 16697, 17218]]),  # int16_1channel
        (2, [[12592, 13620, 16697], [13106, 14134, 17218]]),  # int16_2channel
        (4, [[858927408, 926299444, 1128415545]]),  # int32_1channel
        (4, [[858927408], [926299444], [1128415545]]),  # int32_3channel
    ],
    ids=[
        "int8_1channel",
        "int8_2channel",
        "int8_4channel",
        "int16_1channel",
        "int16_2channel",
        "int32_1channel",
        "int32_3channel",
    ],
)
def test_to_array(setup_data, sample_width, expected):
    data = setup_data
    channels = len(expected)
    expected = np.array(expected)
    result = signal.to_array(data, sample_width, channels)
    assert (result == expected).all()
    assert result.dtype == np.float64
    assert result.shape == expected.shape


@pytest.mark.parametrize(
    "x, aggregation_fn, expected",
    [
        ([300, 320, 400, 600], None, 52.506639194632434),  # mono_simple
        ([0, 0, 0], None, -200),  # mono_zeros
        (
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            None,
            [52.506639194632434, 46.48603928135281],
        ),  # stereo_no_agg
        (
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            np.mean,
            49.49633923799262,
        ),  # stereo_mean_agg
        (
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            min,
            46.48603928135281,
        ),  # stereo_min_agg
        (
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            max,
            52.506639194632434,
        ),  # stereo_max_agg
    ],
    ids=[
        "mono_simple",
        "mono_zeros",
        "stereo_no_agg",
        "mean_agg",
        "stereo_min_agg",
        "stereo_max_agg",
    ],
)
def test_calculate_energy(x, aggregation_fn, expected):
    energy = signal.calculate_energy(x, aggregation_fn)
    assert (energy == expected).all()
