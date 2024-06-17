import math
from array import array as array_
from unittest.mock import patch

import numpy as np
import pytest

from auditok import signal
from auditok.exceptions import TimeFormatError
from auditok.util import (
    AudioEnergyValidator,
    make_channel_selector,
    make_duration_formatter,
)


@pytest.fixture
def setup_data():
    return b"012345679ABC"


@pytest.mark.parametrize(
    "fmt, duration, expected",
    [
        ("%S", 5400, "5400.000"),  # only_seconds
        ("%I", 5400, "5400000"),  # only_millis
        ("%h:%m:%s.%i", 3725.365, "01:02:05.365"),  # full
        ("%h:%m:%s.%i", 1925.075, "00:32:05.075"),  # full_zero_hours
        ("%h:%m:%s.%i", 3659.075, "01:00:59.075"),  # full_zero_minutes
        ("%h:%m:%s.%i", 3720.075, "01:02:00.075"),  # full_zero_seconds
        ("%h:%m:%s.%i", 3725, "01:02:05.000"),  # full_zero_millis
        (
            "%h %h:%m:%s.%i %s",
            3725.365,
            "01 01:02:05.365 05",
        ),  # duplicate_directive
        ("%h:%m:%s", 3725, "01:02:05"),  # no_millis
        ("%h:%m", 3725, "01:02"),  # no_seconds
        ("%h", 3725, "01"),  # no_minutes
        ("%m:%s.%i", 3725, "02:05.000"),  # no_hours
    ],
    ids=[
        "only_seconds",
        "only_millis",
        "full",
        "full_zero_hours",
        "full_zero_minutes",
        "full_zero_seconds",
        "full_zero_millis",
        "duplicate_directive",
        "no_millis",
        "no_seconds",
        "no_minutes",
        "no_hours",
    ],
)
def test_make_duration_formatter(fmt, duration, expected):
    formatter = make_duration_formatter(fmt)
    result = formatter(duration)
    assert result == expected


@pytest.mark.parametrize(
    "fmt",
    [
        "%S %S",  # duplicate_only_seconds
        "%I %I",  # duplicate_only_millis
        "%x",  # unknown_directive
    ],
    ids=[
        "duplicate_only_seconds",
        "duplicate_only_millis",
        "unknown_directive",
    ],
)
def test_make_duration_formatter_error(fmt):
    with pytest.raises(TimeFormatError):
        make_duration_formatter(fmt)


@pytest.mark.parametrize(
    "sample_width, channels, selected, expected",
    [
        (
            1,
            1,
            0,
            [48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67],
        ),  # int8_1channel_select_0
        (1, 2, 0, [48, 50, 52, 54, 57, 66]),  # int8_2channel_select_0
        (1, 3, 0, [48, 51, 54, 65]),  # int8_3channel_select_0
        (1, 3, 1, [49, 52, 55, 66]),  # int8_3channel_select_1
        (1, 3, 2, [50, 53, 57, 67]),  # int8_3channel_select_2
        (1, 4, 0, [48, 52, 57]),  # int8_4channel_select_0
        (
            2,
            1,
            0,
            [12592, 13106, 13620, 14134, 16697, 17218],
        ),  # int16_1channel_select_0
        (2, 2, 0, [12592, 13620, 16697]),  # int16_2channel_select_0
        (2, 2, 1, [13106, 14134, 17218]),  # int16_2channel_select_1
        (2, 3, 0, [12592, 14134]),  # int16_3channel_select_0
        (2, 3, 1, [13106, 16697]),  # int16_3channel_select_1
        (2, 3, 2, [13620, 17218]),  # int16_3channel_select_2
        (
            4,
            1,
            0,
            [858927408, 926299444, 1128415545],
        ),  # int32_1channel_select_0
        (4, 3, 0, [858927408]),  # int32_3channel_select_0
        (4, 3, 1, [926299444]),  # int32_3channel_select_1
        (4, 3, 2, [1128415545]),  # int32_3channel_select_2
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
def test_make_channel_selector_one_channel(
    setup_data, sample_width, channels, selected, expected
):

    selector = make_channel_selector(sample_width, channels, selected)
    result = selector(setup_data)

    dtype = signal.SAMPLE_WIDTH_TO_DTYPE[sample_width]
    expected = np.array(expected).astype(dtype)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "sample_width, channels, selected, expected",
    [
        (
            1,
            1,
            "avg",
            [48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67],
        ),  # int8_1channel
        (1, 2, "mix", [48.5, 50.5, 52.5, 54.5, 61, 66.5]),  # int8_2channel
        (1, 4, "average", [49.5, 53.5, 63.75]),  # int8_4channel
        (
            2,
            1,
            "mix",
            [12592, 13106, 13620, 14134, 16697, 17218],
        ),  # int16_1channel
        (2, 2, "avg", [12849, 13877, 16957.5]),  # int16_2channel
        (4, 3, "average", [971214132.33]),  # int32_3channel
    ],
    ids=[
        "int8_1channel",
        "int8_2channel",
        "int8_4channel",
        "int16_1channel",
        "int16_2channel",
        "int32_3channel",
    ],
)
def test_make_channel_selector_average(
    setup_data, sample_width, channels, selected, expected
):

    selector = make_channel_selector(sample_width, channels, selected)
    result = selector(setup_data).round(2)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "sample_width, channels, selected, expected",
    [
        (
            1,
            1,
            "any",
            [[48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67]],
        ),  # int8_1channel
        (
            1,
            2,
            None,
            [[48, 50, 52, 54, 57, 66], [49, 51, 53, 55, 65, 67]],
        ),  # int8_2channel
        (
            1,
            4,
            "any",
            [[48, 52, 57], [49, 53, 65], [50, 54, 66], [51, 55, 67]],
        ),  # int8_4channel
        (
            2,
            2,
            None,
            [[12592, 13620, 16697], [13106, 14134, 17218]],
        ),  # int16_2channel
        (
            4,
            3,
            "any",
            [[858927408], [926299444], [1128415545]],
        ),  # int32_3channel
    ],
    ids=[
        "int8_1channel",
        "int8_2channel",
        "int8_4channel",
        "int16_2channel",
        "int32_3channel",
    ],
)
def test_make_channel_selector_any(
    setup_data, sample_width, channels, selected, expected
):

    # Use signal functions with numpy implementation
    selector = make_channel_selector(sample_width, channels, selected)
    result = selector(setup_data)
    assert (result == expected).all()


class TestAudioEnergyValidator:
    @pytest.mark.parametrize(
        "data, channels, use_channel, expected",
        [
            ([350, 400], 1, None, True),  # mono_valid_uc_None
            ([350, 400], 1, "any", True),  # mono_valid_uc_any
            ([350, 400], 1, 0, True),  # mono_valid_uc_0
            ([350, 400], 1, "mix", True),  # mono_valid_uc_mix
            ([300, 300], 1, None, False),  # mono_invalid_uc_None
            ([300, 400, 350, 300], 2, None, True),  # stereo_valid_uc_None
            ([300, 400, 350, 300], 2, "any", True),  # stereo_valid_uc_any
            ([300, 400, 350, 300], 2, "mix", True),  # stereo_valid_uc_mix
            ([300, 400, 350, 300], 2, "avg", True),  # stereo_valid_uc_avg
            (
                [300, 400, 300, 300],
                2,
                "average",
                True,
            ),  # stereo_valid_uc_average
            (
                [634, 0, 634, 0],
                2,
                "mix",
                True,
            ),  # stereo_valid_uc_mix_with_null_channel
            ([320, 100, 320, 100], 2, 0, True),  # stereo_valid_uc_0
            ([100, 320, 100, 320], 2, 1, True),  # stereo_valid_uc_1
            ([280, 100, 280, 100], 2, None, False),  # stereo_invalid_uc_None
            ([280, 100, 280, 100], 2, "any", False),  # stereo_invalid_uc_any
            ([400, 200, 400, 200], 2, "mix", False),  # stereo_invalid_uc_mix
            ([300, 400, 300, 400], 2, 0, False),  # stereo_invalid_uc_0
            ([400, 300, 400, 300], 2, 1, False),  # stereo_invalid_uc_1
            ([0, 0, 0, 0], 2, None, False),  # zeros
        ],
        ids=[
            "mono_valid_uc_None",
            "mono_valid_uc_any",
            "mono_valid_uc_0",
            "mono_valid_uc_mix",
            "mono_invalid_uc_None",
            "stereo_valid_uc_None",
            "stereo_valid_uc_any",
            "stereo_valid_uc_mix",
            "stereo_valid_uc_avg",
            "stereo_valid_uc_average",
            "stereo_valid_uc_mix_with_null_channel",
            "stereo_valid_uc_0",
            "stereo_valid_uc_1",
            "stereo_invalid_uc_None",
            "stereo_invalid_uc_any",
            "stereo_invalid_uc_mix",
            "stereo_invalid_uc_0",
            "stereo_invalid_uc_1",
            "zeros",
        ],
    )
    def test_audio_energy_validator(
        self, data, channels, use_channel, expected
    ):

        data = array_("h", data)
        sample_width = 2
        energy_threshold = 50
        validator = AudioEnergyValidator(
            energy_threshold, sample_width, channels, use_channel
        )

        if expected:
            assert validator.is_valid(data)
        else:
            assert not validator.is_valid(data)
