"""
@author: Amine Sehili <amine.sehili@gmail.com>
"""
from array import array
import unittest
from genty import genty, genty_dataset
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


@genty
class TestAudioSource(unittest.TestCase):

    # TODO when use_channel is None, return samples from all channels

    @genty_dataset(
        mono=("mono_400Hz", (400,)),
        multichannel=("3channel_400-800-1600Hz", (400, 800, 1600)),
    )
    def test_BufferAudioSource_read_all(self, file_suffix, frequencies):
        file = "tests/data/test_16KHZ_{}.raw".format(file_suffix)
        with open(file, "rb") as fp:
            expected = fp.read()
        channels = len(frequencies)
        audio_source = BufferAudioSource(expected, 16000, 2, channels)
        audio_source.open()
        data = audio_source.read(None)
        self.assertEqual(data, expected)
        audio_source.rewind()
        data = audio_source.read(-10)
        self.assertEqual(data, expected)
        audio_source.close()

    @genty_dataset(
        mono=("mono_400Hz", (400,)),
        multichannel=("3channel_400-800-1600Hz", (400, 800, 1600)),
    )
    def test_RawAudioSource(self, file_suffix, frequencies):
        file = "tests/data/test_16KHZ_{}.raw".format(file_suffix)
        channels = len(frequencies)
        audio_source = RawAudioSource(file, 16000, 2, channels)
        audio_source.open()
        data_read_all = b"".join(audio_source_read_all_gen(audio_source))
        audio_source.close()
        mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
        fmt = FORMAT[audio_source.sample_width]
        expected = array(fmt, _sample_generator(*mono_channels)).tobytes()

        self.assertEqual(data_read_all, expected)

        # assert read all data with None
        audio_source = RawAudioSource(file, 16000, 2, channels)
        audio_source.open()
        data_read_all = audio_source.read(None)
        audio_source.close()
        self.assertEqual(data_read_all, expected)

        # assert read all data with a negative size
        audio_source = RawAudioSource(file, 16000, 2, channels)
        audio_source.open()
        data_read_all = audio_source.read(-10)
        audio_source.close()
        self.assertEqual(data_read_all, expected)

    @genty_dataset(
        mono=("mono_400Hz", (400,)),
        multichannel=("3channel_400-800-1600Hz", (400, 800, 1600)),
    )
    def test_WaveAudioSource(self, file_suffix, frequencies):
        file = "tests/data/test_16KHZ_{}.wav".format(file_suffix)
        audio_source = WaveAudioSource(file)
        audio_source.open()
        data = b"".join(audio_source_read_all_gen(audio_source))
        audio_source.close()
        mono_channels = [PURE_TONE_DICT[freq] for freq in frequencies]
        fmt = FORMAT[audio_source.sample_width]
        expected = array(fmt, _sample_generator(*mono_channels)).tobytes()

        self.assertEqual(data, expected)

        # assert read all data with None
        audio_source = WaveAudioSource(file)
        audio_source.open()
        data_read_all = audio_source.read(None)
        audio_source.close()
        self.assertEqual(data_read_all, expected)

        # assert read all data with a negative size
        audio_source = WaveAudioSource(file)
        audio_source.open()
        data_read_all = audio_source.read(-10)
        audio_source.close()
        self.assertEqual(data_read_all, expected)


@genty
class TestBufferAudioSource_SR10_SW1_CH1(unittest.TestCase):
    def setUp(self):
        self.data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        self.audio_source = BufferAudioSource(
            data=self.data, sampling_rate=10, sample_width=1, channels=1
        )
        self.audio_source.open()

    def tearDown(self):
        self.audio_source.close()

    def test_sr10_sw1_ch1_read_1(self):
        block = self.audio_source.read(1)
        exp = b"A"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

    def test_sr10_sw1_ch1_read_6(self):
        block = self.audio_source.read(6)
        exp = b"ABCDEF"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

    def test_sr10_sw1_ch1_read_multiple(self):
        block = self.audio_source.read(1)
        exp = b"A"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

        block = self.audio_source.read(6)
        exp = b"BCDEFG"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

        block = self.audio_source.read(13)
        exp = b"HIJKLMNOPQRST"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

        block = self.audio_source.read(9999)
        exp = b"UVWXYZ012345"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

    def test_sr10_sw1_ch1_read_all(self):
        block = self.audio_source.read(9999)
        self.assertEqual(
            block,
            self.data,
            msg="wrong block, expected: {}, found: {} ".format(
                self.data, block
            ),
        )

        block = self.audio_source.read(1)
        self.assertEqual(
            block,
            None,
            msg="wrong block, expected: {}, found: {} ".format(None, block),
        )

    def test_sr10_sw1_ch1_get_sampling_rate(self):
        srate = self.audio_source.get_sampling_rate()
        self.assertEqual(
            srate,
            10,
            msg="wrong sampling rate, expected: 10, found: {0} ".format(srate),
        )

    def test_sr10_sw1_ch1_get_sample_width(self):
        swidth = self.audio_source.get_sample_width()
        self.assertEqual(
            swidth,
            1,
            msg="wrong sample width, expected: 1, found: {0} ".format(swidth),
        )

    def test_sr10_sw1_ch1_get_channels(self):
        channels = self.audio_source.get_channels()
        self.assertEqual(
            channels,
            1,
            msg="wrong number of channels, expected: 1, found: {0} ".format(
                channels
            ),
        )

    @genty_dataset(
        empty=([], 0, 0, 0),
        zero=([0], 0, 0, 0),
        five=([5], 5, 0.5, 500),
        multiple=([5, 20], 25, 2.5, 2500),
    )
    def test_position(
        self, block_sizes, expected_sample, expected_second, expected_ms
    ):
        for block_size in block_sizes:
            self.audio_source.read(block_size)
        position = self.audio_source.position
        self.assertEqual(
            position,
            expected_sample,
            msg="wrong stream position, expected: {}, found: {}".format(
                expected_sample, position
            ),
        )

        position_s = self.audio_source.position_s
        self.assertEqual(
            position_s,
            expected_second,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_second, position_s
            ),
        )

        position_ms = self.audio_source.position_ms
        self.assertEqual(
            position_ms,
            expected_ms,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_ms, position_ms
            ),
        )

    @genty_dataset(
        zero=(0, 0, 0, 0),
        one=(1, 1, 0.1, 100),
        ten=(10, 10, 1, 1000),
        negative_1=(-1, 31, 3.1, 3100),
        negative_2=(-7, 25, 2.5, 2500),
    )
    def test_position_setter(
        self, position, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position = position

        position = self.audio_source.position
        self.assertEqual(
            position,
            expected_sample,
            msg="wrong stream position, expected: {}, found: {}".format(
                expected_sample, position
            ),
        )

        position_s = self.audio_source.position_s
        self.assertEqual(
            position_s,
            expected_second,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_second, position_s
            ),
        )

        position_ms = self.audio_source.position_ms
        self.assertEqual(
            position_ms,
            expected_ms,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_ms, position_ms
            ),
        )

    @genty_dataset(
        zero=(0, 0, 0, 0),
        one=(0.1, 1, 0.1, 100),
        ten=(1, 10, 1, 1000),
        negative_1=(-0.1, 31, 3.1, 3100),
        negative_2=(-0.7, 25, 2.5, 2500),
    )
    def test_position_s_setter(
        self, position_s, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position_s = position_s

        position = self.audio_source.position
        self.assertEqual(
            position,
            expected_sample,
            msg="wrong stream position, expected: {}, found: {}".format(
                expected_sample, position
            ),
        )

        position_s = self.audio_source.position_s
        self.assertEqual(
            position_s,
            expected_second,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_second, position_s
            ),
        )

        position_ms = self.audio_source.position_ms
        self.assertEqual(
            position_ms,
            expected_ms,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_ms, position_ms
            ),
        )

    @genty_dataset(
        zero=(0, 0, 0, 0),
        one=(100, 1, 0.1, 100),
        ten=(1000, 10, 1, 1000),
        negative_1=(-100, 31, 3.1, 3100),
        negative_2=(-700, 25, 2.5, 2500),
    )
    def test_position_ms_setter(
        self, position_ms, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position_ms = position_ms

        position = self.audio_source.position
        self.assertEqual(
            position,
            expected_sample,
            msg="wrong stream position, expected: {}, found: {}".format(
                expected_sample, position
            ),
        )

        position_s = self.audio_source.position_s
        self.assertEqual(
            position_s,
            expected_second,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_second, position_s
            ),
        )

        position_ms = self.audio_source.position_ms
        self.assertEqual(
            position_ms,
            expected_ms,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_ms, position_ms
            ),
        )

    @genty_dataset(positive=((100,)), negative=(-100,))
    def test_position_setter_out_of_range(self, position):
        with self.assertRaises(IndexError):
            self.audio_source.position = position

    @genty_dataset(positive=((100,)), negative=(-100,))
    def test_position_s_setter_out_of_range(self, position_s):
        with self.assertRaises(IndexError):
            self.audio_source.position_s = position_s

    @genty_dataset(positive=((10000,)), negative=(-10000,))
    def test_position_ms_setter_out_of_range(self, position_ms):
        with self.assertRaises(IndexError):
            self.audio_source.position_ms = position_ms

    def test_sr10_sw1_ch1_initial_position_s_0(self):
        tp = self.audio_source.position_s
        self.assertEqual(
            tp,
            0.0,
            msg="wrong time position, expected: 0.0, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_position_s_1_after_read(self):
        srate = self.audio_source.sampling_rate
        # read one second
        self.audio_source.read(srate)
        tp = self.audio_source.position_s
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_position_s_2_5(self):
        # read 2.5 seconds
        self.audio_source.read(25)
        tp = self.audio_source.position_s
        self.assertEqual(
            tp,
            2.5,
            msg="wrong time position, expected: 2.5, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_position_s_0(self):
        self.audio_source.read(10)
        self.audio_source.position_s = 0
        tp = self.audio_source.position_s
        self.assertEqual(
            tp,
            0.0,
            msg="wrong time position, expected: 0.0, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_position_s_1(self):
        self.audio_source.position_s = 1
        tp = self.audio_source.position_s
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_rewind(self):
        self.audio_source.read(10)
        self.audio_source.rewind()
        tp = self.audio_source.position
        self.assertEqual(
            tp, 0, msg="wrong position, expected: 0.0, found: {0} ".format(tp)
        )

    def test_sr10_sw1_ch1_read_closed(self):
        self.audio_source.close()
        with self.assertRaises(Exception):
            self.audio_source.read(1)


@genty
class TestBufferAudioSource_SR16_SW2_CH1(unittest.TestCase):
    def setUp(self):
        self.data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        self.audio_source = BufferAudioSource(
            data=self.data, sampling_rate=16, sample_width=2, channels=1
        )
        self.audio_source.open()

    def tearDown(self):
        self.audio_source.close()

    def test_sr16_sw2_ch1_read_1(self):
        block = self.audio_source.read(1)
        exp = b"AB"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

    def test_sr16_sw2_ch1_read_6(self):
        block = self.audio_source.read(6)
        exp = b"ABCDEFGHIJKL"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

    def test_sr16_sw2_ch1_read_multiple(self):
        block = self.audio_source.read(1)
        exp = b"AB"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

        block = self.audio_source.read(6)
        exp = b"CDEFGHIJKLMN"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

        block = self.audio_source.read(5)
        exp = b"OPQRSTUVWX"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

        block = self.audio_source.read(9999)
        exp = b"YZ012345"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

    def test_sr16_sw2_ch1_read_all(self):
        block = self.audio_source.read(9999)
        self.assertEqual(
            block,
            self.data,
            msg="wrong block, expected: {0}, found: {1} ".format(
                self.data, block
            ),
        )

        block = self.audio_source.read(1)
        self.assertEqual(
            block,
            None,
            msg="wrong block, expected: {0}, found: {1} ".format(None, block),
        )

    def test_sr16_sw2_ch1_get_sampling_rate(self):
        srate = self.audio_source.get_sampling_rate()
        self.assertEqual(
            srate,
            16,
            msg="wrong sampling rate, expected: 10, found: {0} ".format(srate),
        )

    def test_sr16_sw2_ch1_get_sample_width(self):
        swidth = self.audio_source.get_sample_width()
        self.assertEqual(
            swidth,
            2,
            msg="wrong sample width, expected: 1, found: {0} ".format(swidth),
        )

    def test_sr16_sw2_ch1_get_channels(self):

        channels = self.audio_source.get_channels()
        self.assertEqual(
            channels,
            1,
            msg="wrong number of channels, expected: 1, found: {0} ".format(
                channels
            ),
        )

    @genty_dataset(
        empty=([], 0, 0, 0),
        zero=([0], 0, 0, 0),
        two=([2], 2, 2 / 16, int(2000 / 16)),
        eleven=([11], 11, 11 / 16, int(11 * 1000 / 16)),
        multiple=([4, 8], 12, 0.75, 750),
    )
    def test_position(
        self, block_sizes, expected_sample, expected_second, expected_ms
    ):
        for block_size in block_sizes:
            self.audio_source.read(block_size)
        position = self.audio_source.position
        self.assertEqual(
            position,
            expected_sample,
            msg="wrong stream position, expected: {}, found: {}".format(
                expected_sample, position
            ),
        )

        position_s = self.audio_source.position_s
        self.assertEqual(
            position_s,
            expected_second,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_second, position_s
            ),
        )

        position_ms = self.audio_source.position_ms
        self.assertEqual(
            position_ms,
            expected_ms,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_ms, position_ms
            ),
        )

    def test_sr16_sw2_ch1_read_position_0(self):
        self.audio_source.read(10)
        self.audio_source.position = 0
        pos = self.audio_source.position
        self.assertEqual(
            pos, 0, msg="wrong position, expected: 0, found: {0} ".format(pos)
        )

    @genty_dataset(
        zero=(0, 0, 0, 0),
        one=(1, 1, 1 / 16, int(1000 / 16)),
        ten=(10, 10, 10 / 16, int(10000 / 16)),
        negative_1=(-1, 15, 15 / 16, int(15000 / 16)),
        negative_2=(-7, 9, 9 / 16, int(9000 / 16)),
    )
    def test_position_setter(
        self, position, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position = position

        position = self.audio_source.position
        self.assertEqual(
            position,
            expected_sample,
            msg="wrong stream position, expected: {}, found: {}".format(
                expected_sample, position
            ),
        )

        position_s = self.audio_source.position_s
        self.assertEqual(
            position_s,
            expected_second,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_second, position_s
            ),
        )

        position_ms = self.audio_source.position_ms
        self.assertEqual(
            position_ms,
            expected_ms,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_ms, position_ms
            ),
        )

    @genty_dataset(
        zero=(0, 0, 0, 0),
        one=(0.1, 1, 1 / 16, int(1000 / 16)),
        two=(1 / 8, 2, 1 / 8, int(1 / 8 * 1000)),
        twelve=(0.75, 12, 0.75, 750),
        negative_1=(-0.1, 15, 15 / 16, int(15000 / 16)),
        negative_2=(-0.7, 5, 5 / 16, int(5000 / 16)),
    )
    def test_position_s_setter(
        self, position_s, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position_s = position_s

        position = self.audio_source.position
        self.assertEqual(
            position,
            expected_sample,
            msg="wrong stream position, expected: {}, found: {}".format(
                expected_sample, position
            ),
        )

        position_s = self.audio_source.position_s
        self.assertEqual(
            position_s,
            expected_second,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_second, position_s
            ),
        )

        position_ms = self.audio_source.position_ms
        self.assertEqual(
            position_ms,
            expected_ms,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_ms, position_ms
            ),
        )

    @genty_dataset(
        zero=(0, 0, 0, 0),
        one=(100, 1, 1 / 16, int(1000 / 16)),
        ten=(1000, 16, 1, 1000),
        negative_1=(-100, 15, 15 / 16, int(15 * 1000 / 16)),
        negative_2=(-500, 8, 0.5, 500),
        negative_3=(-700, 5, 5 / 16, int(5 * 1000 / 16)),
    )
    def test_position_ms_setter(
        self, position_ms, expected_sample, expected_second, expected_ms
    ):
        self.audio_source.position_ms = position_ms

        position = self.audio_source.position
        self.assertEqual(
            position,
            expected_sample,
            msg="wrong stream position, expected: {}, found: {}".format(
                expected_sample, position
            ),
        )

        position_s = self.audio_source.position_s
        self.assertEqual(
            position_s,
            expected_second,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_second, position_s
            ),
        )

        position_ms = self.audio_source.position_ms
        self.assertEqual(
            position_ms,
            expected_ms,
            msg="wrong stream position_s, expected: {}, found: {}".format(
                expected_ms, position_ms
            ),
        )

    def test_sr16_sw2_ch1_rewind(self):
        self.audio_source.read(10)
        self.audio_source.rewind()
        tp = self.audio_source.position
        self.assertEqual(
            tp, 0, msg="wrong position, expected: 0.0, found: {0} ".format(tp)
        )


class TestBufferAudioSource_SR11_SW4_CH1(unittest.TestCase):
    def setUp(self):
        self.data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefgh"
        self.audio_source = BufferAudioSource(
            data=self.data, sampling_rate=11, sample_width=4, channels=1
        )
        self.audio_source.open()

    def tearDown(self):
        self.audio_source.close()

    def test_sr11_sw4_ch1_read_1(self):
        block = self.audio_source.read(1)
        exp = b"ABCD"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

    def test_sr11_sw4_ch1_read_6(self):
        block = self.audio_source.read(6)
        exp = b"ABCDEFGHIJKLMNOPQRSTUVWX"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

    def test_sr11_sw4_ch1_read_multiple(self):
        block = self.audio_source.read(1)
        exp = b"ABCD"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

        block = self.audio_source.read(6)
        exp = b"EFGHIJKLMNOPQRSTUVWXYZ01"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

        block = self.audio_source.read(3)
        exp = b"23456789abcd"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

        block = self.audio_source.read(9999)
        exp = b"efgh"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

    def test_sr11_sw4_ch1_read_all(self):
        block = self.audio_source.read(9999)
        self.assertEqual(
            block,
            self.data,
            msg="wrong block, expected: {0}, found: {1} ".format(
                self.data, block
            ),
        )

        block = self.audio_source.read(1)
        self.assertEqual(
            block,
            None,
            msg="wrong block, expected: {0}, found: {1} ".format(None, block),
        )

    def test_sr11_sw4_ch1_get_sampling_rate(self):
        srate = self.audio_source.get_sampling_rate()
        self.assertEqual(
            srate,
            11,
            msg="wrong sampling rate, expected: 10, found: {0} ".format(srate),
        )

    def test_sr11_sw4_ch1_get_sample_width(self):
        swidth = self.audio_source.get_sample_width()
        self.assertEqual(
            swidth,
            4,
            msg="wrong sample width, expected: 1, found: {0} ".format(swidth),
        )

    def test_sr11_sw4_ch1_get_channels(self):
        channels = self.audio_source.get_channels()
        self.assertEqual(
            channels,
            1,
            msg="wrong number of channels, expected: 1, found: {0} ".format(
                channels
            ),
        )

    def test_sr11_sw4_ch1_intial_position_0(self):
        pos = self.audio_source.position
        self.assertEqual(
            pos, 0, msg="wrong position, expected: 0, found: {0} ".format(pos)
        )

    def test_sr11_sw4_ch1_position_5(self):
        self.audio_source.read(5)
        pos = self.audio_source.position
        self.assertEqual(
            pos, 5, msg="wrong position, expected: 5, found: {0} ".format(pos)
        )

    def test_sr11_sw4_ch1_position_9(self):
        self.audio_source.read(5)
        self.audio_source.read(4)
        pos = self.audio_source.position
        self.assertEqual(
            pos, 9, msg="wrong position, expected: 5, found: {0} ".format(pos)
        )

    def test_sr11_sw4_ch1_position_0(self):
        self.audio_source.read(10)
        self.audio_source.position = 0
        pos = self.audio_source.position
        self.assertEqual(
            pos, 0, msg="wrong position, expected: 0, found: {0} ".format(pos)
        )

    def test_sr11_sw4_ch1_position_10(self):
        self.audio_source.position = 10
        pos = self.audio_source.position
        self.assertEqual(
            pos,
            10,
            msg="wrong position, expected: 10, found: {0} ".format(pos),
        )

    def test_sr11_sw4_ch1_initial_position_s_0(self):
        tp = self.audio_source.position_s
        self.assertEqual(
            tp,
            0.0,
            msg="wrong time position, expected: 0.0, found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_position_s_1_after_read(self):
        srate = self.audio_source.sampling_rate
        # read one second
        self.audio_source.read(srate)
        tp = self.audio_source.position_s
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_position_s_0_63(self):
        # read 2.5 seconds
        self.audio_source.read(7)
        tp = self.audio_source.position_s
        self.assertAlmostEqual(
            tp,
            0.636363636364,
            msg="wrong time position, expected: 0.636363636364, "
            "found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_position_s_0(self):
        self.audio_source.read(10)
        self.audio_source.position_s = 0
        tp = self.audio_source.position_s
        self.assertEqual(
            tp,
            0.0,
            msg="wrong time position, expected: 0.0, found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_position_s_1(self):
        self.audio_source.position_s = 1
        tp = self.audio_source.position_s
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_rewind(self):
        self.audio_source.read(10)
        self.audio_source.rewind()
        tp = self.audio_source.position
        self.assertEqual(
            tp, 0, msg="wrong position, expected: 0.0, found: {0} ".format(tp)
        )


class TestBufferAudioSourceCreationException(unittest.TestCase):
    def test_wrong_sample_width_value(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            _ = BufferAudioSource(
                data=b"ABCDEFGHI", sampling_rate=9, sample_width=3, channels=1
            )
        self.assertEqual(
            "Sample width must be one of: 1, 2 or 4 (bytes)",
            str(audio_param_err.exception),
        )

    def test_wrong_data_buffer_size(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            _ = BufferAudioSource(
                data=b"ABCDEFGHI", sampling_rate=8, sample_width=2, channels=1
            )
        self.assertEqual(
            "The length of audio data must be an integer "
            "multiple of `sample_width * channels`",
            str(audio_param_err.exception),
        )


class TestAudioSourceProperties(unittest.TestCase):
    def test_read_properties(self):
        data = b""
        sampling_rate = 8000
        sample_width = 2
        channels = 1
        a_source = BufferAudioSource(
            data, sampling_rate, sample_width, channels
        )

        self.assertEqual(a_source.sampling_rate, sampling_rate)
        self.assertEqual(a_source.sample_width, sample_width)
        self.assertEqual(a_source.channels, channels)

    def test_set_readonly_properties_exception(self):
        data = b""
        sampling_rate = 8000
        sample_width = 2
        channels = 1
        a_source = BufferAudioSource(
            data, sampling_rate, sample_width, channels
        )

        with self.assertRaises(AttributeError):
            a_source.sampling_rate = 16000
            a_source.sample_width = 1
            a_source.channels = 2


class TestAudioSourceShortProperties(unittest.TestCase):
    def test_read_short_properties(self):
        data = b""
        sampling_rate = 8000
        sample_width = 2
        channels = 1
        a_source = BufferAudioSource(
            data, sampling_rate, sample_width, channels
        )

        self.assertEqual(a_source.sr, sampling_rate)
        self.assertEqual(a_source.sw, sample_width)
        self.assertEqual(a_source.ch, channels)

    def test_set_readonly_short_properties_exception(self):
        data = b""
        sampling_rate = 8000
        sample_width = 2
        channels = 1
        a_source = BufferAudioSource(
            data, sampling_rate, sample_width, channels
        )

        with self.assertRaises(AttributeError):
            a_source.sr = 16000
            a_source.sw = 1
            a_source.ch = 2


if __name__ == "__main__":
    unittest.main()
