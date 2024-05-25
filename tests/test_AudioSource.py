"""
@author: Amine Sehili <amine.sehili@gmail.com>
"""

from array import array
import pytest
from auditok.io import (
    AudioParameterError,
    BufferAudioSource,
    RawAudioSource,
    WaveAudioSource,
)
from auditok.signal import FORMAT
from test_util import PURE_TONE_DICT, _sample_generator


def audio_source_read_all_gen(audio_source, size=None):
    if size is None:
        size = int(audio_source.sr * 0.1)  # 100ms
    while True:
        data = audio_source.read(size)
        if data is None:
            break
        yield data


@pytest.mark.parametrize(
    "file_suffix, frequencies",
    [
        ("mono_400Hz", (400,)),  # mono
        ("3channel_400-800-1600Hz", (400, 800, 1600)),  # multichannel
    ],
    ids=["mono", "multichannel"],
)
def test_BufferAudioSource_read_all(file_suffix, frequencies):
    file = "tests/data/test_16KHZ_{}.raw".format(file_suffix)
    with open(file, "rb") as fp:
        expected = fp.read()
    channels = len(frequencies)
    audio_source = BufferAudioSource(expected, 16000, 2, channels)
    audio_source.open()
    data = audio_source.read(None)
    assert data == expected
    audio_source.rewind()
    data = audio_source.read(-10)
    assert data == expected
    audio_source.close()


@pytest.mark.parametrize(
    "file_suffix, frequencies",
    [
        ("mono_400Hz", (400,)),  # mono
        ("3channel_400-800-1600Hz", (400, 800, 1600)),  # multichannel
    ],
    ids=["mono", "multichannel"],
)
def test_RawAudioSource(file_suffix, frequencies):
    file = "tests/data/test_16KHZ_{}.raw".format(file_suffix)
    channels = len(frequencies)
    audio_source = RawAudioSource(file, 16000, 2, channels)
    audio_source.open()
    data_read_all = b"".join(audio_source_read_all_gen(audio_source))
    audio_source.close()
    mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
    fmt = FORMAT[audio_source.sample_width]
    expected = array(fmt, _sample_generator(*mono_channels)).tobytes()

    assert data_read_all == expected

    # assert read all data with None
    audio_source = RawAudioSource(file, 16000, 2, channels)
    audio_source.open()
    data_read_all = audio_source.read(None)
    audio_source.close()
    assert data_read_all == expected

    # assert read all data with a negative size
    audio_source = RawAudioSource(file, 16000, 2, channels)
    audio_source.open()
    data_read_all = audio_source.read(-10)
    audio_source.close()
    assert data_read_all == expected


@pytest.mark.parametrize(
    "file_suffix, frequencies",
    [
        ("mono_400Hz", (400,)),  # mono
        ("3channel_400-800-1600Hz", (400, 800, 1600)),  # multichannel
    ],
    ids=["mono", "multichannel"],
)
def test_WaveAudioSource(file_suffix, frequencies):
    file = "tests/data/test_16KHZ_{}.wav".format(file_suffix)
    audio_source = WaveAudioSource(file)
    audio_source.open()
    data = b"".join(audio_source_read_all_gen(audio_source))
    audio_source.close()
    mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
    fmt = FORMAT[audio_source.sample_width]
    expected = array(fmt, _sample_generator(*mono_channels)).tobytes()

    assert data == expected

    # assert read all data with None
    audio_source = WaveAudioSource(file)
    audio_source.open()
    data_read_all = audio_source.read(None)
    audio_source.close()
    assert data_read_all == expected

    # assert read all data with a negative size
    audio_source = WaveAudioSource(file)
    audio_source.open()
    data_read_all = audio_source.read(-10)
    audio_source.close()
    assert data_read_all == expected


class TestBufferAudioSource_SR10_SW1_CH1:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        self.audio_source = BufferAudioSource(
            data=self.data, sampling_rate=10, sample_width=1, channels=1
        )
        self.audio_source.open()
        yield
        self.audio_source.close()

    def test_sr10_sw1_ch1_read_1(self):
        block = self.audio_source.read(1)
        exp = b"A"
        assert block == exp

    def test_sr10_sw1_ch1_read_6(self):
        block = self.audio_source.read(6)
        exp = b"ABCDEF"
        assert block == exp

    def test_sr10_sw1_ch1_read_multiple(self):
        block = self.audio_source.read(1)
        exp = b"A"
        assert block == exp

        block = self.audio_source.read(6)
        exp = b"BCDEFG"
        assert block == exp

        block = self.audio_source.read(13)
        exp = b"HIJKLMNOPQRST"
        assert block == exp

        block = self.audio_source.read(9999)
        exp = b"UVWXYZ012345"
        assert block == exp

    def test_sr10_sw1_ch1_read_all(self):
        block = self.audio_source.read(9999)
        assert block == self.data

        block = self.audio_source.read(1)
        assert block is None

    def test_sr10_sw1_ch1_sampling_rate(self):
        srate = self.audio_source.sampling_rate
        assert srate == 10

    def test_sr10_sw1_ch1_sample_width(self):
        swidth = self.audio_source.sample_width
        assert swidth == 1

    def test_sr10_sw1_ch1_channels(self):
        channels = self.audio_source.channels
        assert channels == 1

    @pytest.mark.parametrize(
        "block_sizes, expected_sample, expected_second, expected_ms",
        [
            ([], 0, 0, 0),  # empty
            ([0], 0, 0, 0),  # zero
            ([5], 5, 0.5, 500),  # five
            ([5, 20], 25, 2.5, 2500),  # multiple
        ],
        ids=["empty", "zero", "five", "multiple"],
    )
    def test_position(
        self, block_sizes, expected_sample, expected_second, expected_ms
    ):
        for block_size in block_sizes:
            self.audio_source.read(block_size)
        position = self.audio_source.position
        assert position == expected_sample

        position_s = self.audio_source.position_s
        assert position_s == expected_second

        position_ms = self.audio_source.position_ms
        assert position_ms == expected_ms

    @pytest.mark.parametrize(
        "position, expected_sample, expected_second, expected_ms",
        [
            (0, 0, 0, 0),  # zero
            (1, 1, 0.1, 100),  # one
            (10, 10, 1, 1000),  # ten
            (-1, 31, 3.1, 3100),  # negative_1
            (-7, 25, 2.5, 2500),  # negative_2
        ],
        ids=["zero", "one", "ten", "negative_1", "negative_2"],
    )
    def test_position_setter(
        self, position, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position = position

        position = self.audio_source.position
        assert position == expected_sample

        position_s = self.audio_source.position_s
        assert position_s == expected_second

        position_ms = self.audio_source.position_ms
        assert position_ms == expected_ms

    @pytest.mark.parametrize(
        "position_s, expected_sample, expected_second, expected_ms",
        [
            (0, 0, 0, 0),  # zero
            (0.1, 1, 0.1, 100),  # one
            (1, 10, 1, 1000),  # ten
            (-0.1, 31, 3.1, 3100),  # negative_1
            (-0.7, 25, 2.5, 2500),  # negative_2
        ],
        ids=["zero", "one", "ten", "negative_1", "negative_2"],
    )
    def test_position_s_setter(
        self, position_s, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position_s = position_s

        position = self.audio_source.position
        assert position == expected_sample

        position_s = self.audio_source.position_s
        assert position_s == expected_second

        position_ms = self.audio_source.position_ms
        assert position_ms == expected_ms

    @pytest.mark.parametrize(
        "position_ms, expected_sample, expected_second, expected_ms",
        [
            (0, 0, 0, 0),  # zero
            (100, 1, 0.1, 100),  # one
            (1000, 10, 1, 1000),  # ten
            (-100, 31, 3.1, 3100),  # negative_1
            (-700, 25, 2.5, 2500),  # negative_2
        ],
        ids=["zero", "one", "ten", "negative_1", "negative_2"],
    )
    def test_position_ms_setter(
        self, position_ms, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position_ms = position_ms

        position = self.audio_source.position
        assert position == expected_sample

        position_s = self.audio_source.position_s
        assert position_s == expected_second

        position_ms = self.audio_source.position_ms
        assert position_ms == expected_ms

    @pytest.mark.parametrize(
        "position",
        [
            100,  # positive
            -100,  # negative
        ],
        ids=["positive", "negative"],
    )
    def test_position_setter_out_of_range(self, position):
        with pytest.raises(IndexError):
            self.audio_source.position = position

    @pytest.mark.parametrize(
        "position_s",
        [
            100,  # positive
            -100,  # negative
        ],
        ids=["positive", "negative"],
    )
    def test_position_s_setter_out_of_range(self, position_s):
        with pytest.raises(IndexError):
            self.audio_source.position_s = position_s

    @pytest.mark.parametrize(
        "position_ms",
        [
            10000,  # positive
            -10000,  # negative
        ],
        ids=["positive", "negative"],
    )
    def test_position_ms_setter_out_of_range(self, position_ms):
        with pytest.raises(IndexError):
            self.audio_source.position_ms = position_ms

    def test_sr10_sw1_ch1_initial_position_s_0(self):
        tp = self.audio_source.position_s
        assert tp == 0.0

    def test_sr10_sw1_ch1_position_s_1_after_read(self):
        srate = self.audio_source.sampling_rate
        # read one second
        self.audio_source.read(srate)
        tp = self.audio_source.position_s
        assert tp == 1.0

    def test_sr10_sw1_ch1_position_s_2_5(self):
        # read 2.5 seconds
        self.audio_source.read(25)
        tp = self.audio_source.position_s
        assert tp == 2.5

    def test_sr10_sw1_ch1_position_s_0(self):
        self.audio_source.read(10)
        self.audio_source.position_s = 0
        tp = self.audio_source.position_s
        assert tp == 0.0

    def test_sr10_sw1_ch1_position_s_1(self):
        self.audio_source.position_s = 1
        tp = self.audio_source.position_s
        assert tp == 1.0

    def test_sr10_sw1_ch1_rewind(self):
        self.audio_source.read(10)
        self.audio_source.rewind()
        tp = self.audio_source.position
        assert tp == 0

    def test_sr10_sw1_ch1_read_closed(self):
        self.audio_source.close()
        with pytest.raises(Exception):
            self.audio_source.read(1)


class TestBufferAudioSource_SR16_SW2_CH1:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        self.audio_source = BufferAudioSource(
            data=self.data, sampling_rate=16, sample_width=2, channels=1
        )
        self.audio_source.open()
        yield
        self.audio_source.close()

    def test_sr16_sw2_ch1_read_1(self):
        block = self.audio_source.read(1)
        exp = b"AB"
        assert block == exp

    def test_sr16_sw2_ch1_read_6(self):
        block = self.audio_source.read(6)
        exp = b"ABCDEFGHIJKL"
        assert block == exp

    def test_sr16_sw2_ch1_read_multiple(self):
        block = self.audio_source.read(1)
        exp = b"AB"
        assert block == exp

        block = self.audio_source.read(6)
        exp = b"CDEFGHIJKLMN"
        assert block == exp

        block = self.audio_source.read(5)
        exp = b"OPQRSTUVWX"
        assert block == exp

        block = self.audio_source.read(9999)
        exp = b"YZ012345"
        assert block == exp

    def test_sr16_sw2_ch1_read_all(self):
        block = self.audio_source.read(9999)
        assert block == self.data

        block = self.audio_source.read(1)
        assert block is None

    def test_sr16_sw2_ch1_sampling_rate(self):
        srate = self.audio_source.sampling_rate
        assert srate == 16

    def test_sr16_sw2_ch1_sample_width(self):
        swidth = self.audio_source.sample_width
        assert swidth == 2

    def test_sr16_sw2_ch1_channels(self):
        channels = self.audio_source.channels
        assert channels == 1

    @pytest.mark.parametrize(
        "block_sizes, expected_sample, expected_second, expected_ms",
        [
            ([], 0, 0, 0),  # empty
            ([0], 0, 0, 0),  # zero
            ([2], 2, 2 / 16, int(2000 / 16)),  # two
            ([11], 11, 11 / 16, int(11 * 1000 / 16)),  # eleven
            ([4, 8], 12, 0.75, 750),  # multiple
        ],
        ids=["empty", "zero", "two", "eleven", "multiple"],
    )
    def test_position(
        self, block_sizes, expected_sample, expected_second, expected_ms
    ):
        for block_size in block_sizes:
            self.audio_source.read(block_size)
        position = self.audio_source.position
        assert position == expected_sample

        position_s = self.audio_source.position_s
        assert position_s == expected_second

        position_ms = self.audio_source.position_ms
        assert position_ms == expected_ms

    def test_sr16_sw2_ch1_read_position_0(self):
        self.audio_source.read(10)
        self.audio_source.position = 0
        pos = self.audio_source.position
        assert pos == 0

    @pytest.mark.parametrize(
        "position, expected_sample, expected_second, expected_ms",
        [
            (0, 0, 0, 0),  # zero
            (1, 1, 1 / 16, int(1000 / 16)),  # one
            (10, 10, 10 / 16, int(10000 / 16)),  # ten
            (-1, 15, 15 / 16, int(15000 / 16)),  # negative_1
            (-7, 9, 9 / 16, int(9000 / 16)),  # negative_2
        ],
        ids=["zero", "one", "ten", "negative_1", "negative_2"],
    )
    def test_position_setter(
        self, position, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position = position

        position = self.audio_source.position
        assert position == expected_sample

        position_s = self.audio_source.position_s
        assert position_s == expected_second

        position_ms = self.audio_source.position_ms
        assert position_ms == expected_ms

    @pytest.mark.parametrize(
        "position_s, expected_sample, expected_second, expected_ms",
        [
            (0, 0, 0, 0),  # zero
            (0.1, 1, 1 / 16, int(1000 / 16)),  # one
            (1 / 8, 2, 1 / 8, int(1 / 8 * 1000)),  # two
            (0.75, 12, 0.75, 750),  # twelve
            (-0.1, 15, 15 / 16, int(15000 / 16)),  # negative_1
            (-0.7, 5, 5 / 16, int(5000 / 16)),  # negative_2
        ],
        ids=["zero", "one", "two", "twelve", "negative_1", "negative_2"],
    )
    def test_position_s_setter(
        self, position_s, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position_s = position_s

        position = self.audio_source.position
        assert position == expected_sample

        position_s = self.audio_source.position_s
        assert position_s == expected_second

        position_ms = self.audio_source.position_ms
        assert position_ms == expected_ms

    @pytest.mark.parametrize(
        "position_ms, expected_sample, expected_second, expected_ms",
        [
            (0, 0, 0, 0),  # zero
            (100, 1, 1 / 16, int(1000 / 16)),  # one
            (1000, 16, 1, 1000),  # ten
            (-100, 15, 15 / 16, int(15 * 1000 / 16)),  # negative_1
            (-500, 8, 0.5, 500),  # negative_2
            (-700, 5, 5 / 16, int(5 * 1000 / 16)),  # negative_3
        ],
        ids=["zero", "one", "ten", "negative_1", "negative_2", "negative_3"],
    )
    def test_position_ms_setter(
        self, position_ms, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position_ms = position_ms

        position = self.audio_source.position
        assert position == expected_sample

        position_s = self.audio_source.position_s
        assert position_s == expected_second

        position_ms = self.audio_source.position_ms
        assert position_ms == expected_ms

    def test_sr16_sw2_ch1_rewind(self):
        self.audio_source.read(10)
        self.audio_source.rewind()
        tp = self.audio_source.position
        assert tp == 0


class TestBufferAudioSource_SR11_SW4_CH1:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefgh"
        self.audio_source = BufferAudioSource(
            data=self.data, sampling_rate=11, sample_width=4, channels=1
        )
        self.audio_source.open()
        yield
        self.audio_source.close()

    def test_sr11_sw4_ch1_read_1(self):
        block = self.audio_source.read(1)
        exp = b"ABCD"
        assert block == exp

    def test_sr11_sw4_ch1_read_6(self):
        block = self.audio_source.read(6)
        exp = b"ABCDEFGHIJKLMNOPQRSTUVWX"
        assert block == exp

    def test_sr11_sw4_ch1_read_multiple(self):
        block = self.audio_source.read(1)
        exp = b"ABCD"
        assert block == exp

        block = self.audio_source.read(6)
        exp = b"EFGHIJKLMNOPQRSTUVWXYZ01"
        assert block == exp

        block = self.audio_source.read(3)
        exp = b"23456789abcd"
        assert block == exp

        block = self.audio_source.read(9999)
        exp = b"efgh"
        assert block == exp

    def test_sr11_sw4_ch1_read_all(self):
        block = self.audio_source.read(9999)
        assert block == self.data

        block = self.audio_source.read(1)
        assert block is None

    def test_sr11_sw4_ch1_sampling_rate(self):
        srate = self.audio_source.sampling_rate
        assert srate == 11

    def test_sr11_sw4_ch1_sample_width(self):
        swidth = self.audio_source.sample_width
        assert swidth == 4

    def test_sr11_sw4_ch1_channels(self):
        channels = self.audio_source.channels
        assert channels == 1

    def test_sr11_sw4_ch1_intial_position_0(self):
        pos = self.audio_source.position
        assert pos == 0

    def test_sr11_sw4_ch1_position_5(self):
        self.audio_source.read(5)
        pos = self.audio_source.position
        assert pos == 5

    def test_sr11_sw4_ch1_position_9(self):
        self.audio_source.read(5)
        self.audio_source.read(4)
        pos = self.audio_source.position
        assert pos == 9

    def test_sr11_sw4_ch1_position_0(self):
        self.audio_source.read(10)
        self.audio_source.position = 0
        pos = self.audio_source.position
        assert pos == 0

    def test_sr11_sw4_ch1_position_10(self):
        self.audio_source.position = 10
        pos = self.audio_source.position
        assert pos == 10

    def test_sr11_sw4_ch1_initial_position_s_0(self):
        tp = self.audio_source.position_s
        assert tp == 0.0

    def test_sr11_sw4_ch1_position_s_1_after_read(self):
        srate = self.audio_source.sampling_rate
        # read one second
        self.audio_source.read(srate)
        tp = self.audio_source.position_s
        assert tp == 1.0

    def test_sr11_sw4_ch1_position_s_0_63(self):
        # read 2.5 seconds
        self.audio_source.read(7)
        tp = self.audio_source.position_s
        assert tp, pytest.approx(0.636363636364)

    def test_sr11_sw4_ch1_position_s_0(self):
        self.audio_source.read(10)
        self.audio_source.position_s = 0
        tp = self.audio_source.position_s
        assert tp == 0.0

    def test_sr11_sw4_ch1_position_s_1(self):
        self.audio_source.position_s = 1
        tp = self.audio_source.position_s
        assert tp == 1.0

    def test_sr11_sw4_ch1_rewind(self):
        self.audio_source.read(10)
        self.audio_source.rewind()
        tp = self.audio_source.position
        assert tp == 0


class TestBufferAudioSourceCreationException:
    def test_wrong_sample_width_value(self):
        with pytest.raises(AudioParameterError) as audio_param_err:
            _ = BufferAudioSource(
                data=b"ABCDEFGHI", sampling_rate=9, sample_width=3, channels=1
            )
        assert (
            str(audio_param_err.value)
            == "Sample width must be one of: 1, 2 or 4 (bytes)"
        )

    def test_wrong_data_buffer_size(self):
        with pytest.raises(AudioParameterError) as audio_param_err:
            _ = BufferAudioSource(
                data=b"ABCDEFGHI", sampling_rate=8, sample_width=2, channels=1
            )
        assert (
            str(audio_param_err.value)
            == "The length of audio data must be an integer multiple of `sample_width * channels`"
        )


class TestAudioSourceProperties:
    def test_read_properties(self):
        data = b""
        sampling_rate = 8000
        sample_width = 2
        channels = 1
        a_source = BufferAudioSource(
            data, sampling_rate, sample_width, channels
        )

        assert a_source.sampling_rate == sampling_rate
        assert a_source.sample_width == sample_width
        assert a_source.channels == channels

    def test_set_readonly_properties_exception(self):
        data = b""
        sampling_rate = 8000
        sample_width = 2
        channels = 1
        a_source = BufferAudioSource(
            data, sampling_rate, sample_width, channels
        )

        with pytest.raises(AttributeError):
            a_source.sampling_rate = 16000
            a_source.sample_width = 1
            a_source.channels = 2


class TestAudioSourceShortProperties:
    def test_read_short_properties(self):
        data = b""
        sampling_rate = 8000
        sample_width = 2
        channels = 1
        a_source = BufferAudioSource(
            data, sampling_rate, sample_width, channels
        )

        assert a_source.sr == sampling_rate
        assert a_source.sw == sample_width
        assert a_source.ch == channels

    def test_set_readonly_short_properties_exception(self):
        data = b""
        sampling_rate = 8000
        sample_width = 2
        channels = 1
        a_source = BufferAudioSource(
            data, sampling_rate, sample_width, channels
        )

        with pytest.raises(AttributeError):
            a_source.sr = 16000
            a_source.sw = 1
            a_source.ch = 2
