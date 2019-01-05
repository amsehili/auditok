"""
@author: Amine Sehili <amine.sehili@gmail.com>
"""
import unittest
from auditok import BufferAudioSource, AudioParameterError


class TestBufferAudioSource_SR10_SW1_CH1(unittest.TestCase):
    def setUp(self):
        self.data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        self.audio_source = BufferAudioSource(
            data_buffer=self.data, sampling_rate=10, sample_width=1, channels=1
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

    def test_sr10_sw1_ch1_get_position_0(self):
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 0, msg="wrong position, expected: 0, found: {0} ".format(pos)
        )

    def test_sr10_sw1_ch1_get_position_5(self):
        self.audio_source.read(5)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 5, msg="wrong position, expected: 5, found: {0} ".format(pos)
        )

    def test_sr10_sw1_ch1_get_position_25(self):
        self.audio_source.read(5)
        self.audio_source.read(20)

        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 25, msg="wrong position, expected: 5, found: {0} ".format(pos)
        )

    def test_sr10_sw1_ch1_set_position_0(self):
        self.audio_source.read(10)
        self.audio_source.set_position(0)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 0, msg="wrong position, expected: 0, found: {0} ".format(pos)
        )

    def test_sr10_sw1_ch1_set_position_10(self):
        self.audio_source.set_position(10)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos,
            10,
            msg="wrong position, expected: 10, found: {0} ".format(pos),
        )

    def test_sr10_sw1_ch1_get_time_position_0(self):
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            0.0,
            msg="wrong time position, expected: 0.0, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_get_time_position_1(self):
        srate = self.audio_source.get_sampling_rate()
        # read one second
        self.audio_source.read(srate)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_get_time_position_2_5(self):
        # read 2.5 seconds
        self.audio_source.read(25)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            2.5,
            msg="wrong time position, expected: 2.5, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_set_time_position_0(self):
        self.audio_source.read(10)
        self.audio_source.set_time_position(0)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            0.0,
            msg="wrong time position, expected: 0.0, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_set_time_position_1(self):
        self.audio_source.set_time_position(1)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_set_time_position_end(self):
        self.audio_source.set_time_position(100)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            3.2,
            msg="wrong time position, expected: 3.2, found: {0} ".format(tp),
        )

    def test_sr10_sw1_ch1_rewind(self):
        self.audio_source.read(10)
        self.audio_source.rewind()
        tp = self.audio_source.get_position()
        self.assertEqual(
            tp, 0, msg="wrong position, expected: 0.0, found: {0} ".format(tp)
        )

    def test_sr10_sw1_ch1_set_data(self):
        self.audio_source.set_data(b"12345")
        block = self.audio_source.read(9999)
        self.assertEqual(
            block,
            b"12345",
            msg="wrong block, expected: '12345', found: {0} ".format(block),
        )

    def test_sr10_sw1_ch1_read_closed(self):
        self.audio_source.close()
        with self.assertRaises(Exception):
            self.audio_source.read(1)


class TestBufferAudioSource_SR16_SW2_CH1(unittest.TestCase):
    def setUp(self):
        self.data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        self.audio_source = BufferAudioSource(
            data_buffer=self.data, sampling_rate=16, sample_width=2, channels=1
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

    def test_sr16_sw2_ch1_get_position_0(self):
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 0, msg="wrong position, expected: 0, found: {0} ".format(pos)
        )

    def test_sr16_sw2_ch1_get_position_5(self):
        self.audio_source.read(5)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 5, msg="wrong position, expected: 5, found: {0} ".format(pos)
        )

    def test_sr16_sw2_ch1_get_position_15(self):
        self.audio_source.read(5)
        self.audio_source.read(10)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 15, msg="wrong position, expected: 5, found: {0} ".format(pos)
        )

    def test_sr16_sw2_ch1_set_position_0(self):
        self.audio_source.read(10)
        self.audio_source.set_position(0)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 0, msg="wrong position, expected: 0, found: {0} ".format(pos)
        )

    def test_sr16_sw2_ch1_set_position_10(self):
        self.audio_source.set_position(10)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos,
            10,
            msg="wrong position, expected: 10, found: {0} ".format(pos),
        )

    def test_sr16_sw2_ch1_get_time_position_0(self):
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            0.0,
            msg="wrong time position, expected: 0.0, found: {0} ".format(tp),
        )

    def test_sr16_sw2_ch1_get_time_position_1(self):
        srate = self.audio_source.get_sampling_rate()
        # read one second
        self.audio_source.read(srate)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr16_sw2_ch1_get_time_position_0_75(self):
        # read 2.5 seconds
        self.audio_source.read(12)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            0.75,
            msg="wrong time position, expected: 0.75, found: {0} ".format(tp),
        )

    def test_sr16_sw2_ch1_set_time_position_0(self):
        self.audio_source.read(10)
        self.audio_source.set_time_position(0)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            0.0,
            msg="wrong time position, expected: 0.0, found: {0} ".format(tp),
        )

    def test_sr16_sw2_ch1_set_time_position_1(self):
        self.audio_source.set_time_position(1)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr16_sw2_ch1_set_time_position_end(self):
        self.audio_source.set_time_position(100)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr16_sw2_ch1_rewind(self):
        self.audio_source.read(10)
        self.audio_source.rewind()
        tp = self.audio_source.get_position()
        self.assertEqual(
            tp, 0, msg="wrong position, expected: 0.0, found: {0} ".format(tp)
        )

    def test_sr16_sw2_ch1_set_data(self):
        self.audio_source.set_data(b"abcdef")
        block = self.audio_source.read(9999)
        self.assertEqual(
            block,
            b"abcdef",
            msg="wrong block, expected: 'abcdef', found: {0} ".format(block),
        )

    def test_sr16_sw2_ch1_set_data_exception(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            self.audio_source.set_data("abcde")
            self.assertEqual(
                "The length of audio data must be an integer "
                "multiple of `sample_width * channels`",
                str(audio_param_err.exception),
            )

    def test_sr16_sw2_ch1_append_data_exception(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            self.audio_source.append_data("abcde")
            self.assertEqual(
                "The length of audio data must be an integer "
                "multiple of `sample_width * channels`",
                str(audio_param_err.exception),
            )


class TestBufferAudioSource_SR11_SW4_CH1(unittest.TestCase):
    def setUp(self):
        self.data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefgh"
        self.audio_source = BufferAudioSource(
            data_buffer=self.data, sampling_rate=11, sample_width=4, channels=1
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

    def test_sr11_sw4_ch1_get_position_0(self):
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 0, msg="wrong position, expected: 0, found: {0} ".format(pos)
        )

    def test_sr11_sw4_ch1_get_position_5(self):
        self.audio_source.read(5)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 5, msg="wrong position, expected: 5, found: {0} ".format(pos)
        )

    def test_sr11_sw4_ch1_get_position_9(self):
        self.audio_source.read(5)
        self.audio_source.read(4)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 9, msg="wrong position, expected: 5, found: {0} ".format(pos)
        )

    def test_sr11_sw4_ch1_set_position_0(self):
        self.audio_source.read(10)
        self.audio_source.set_position(0)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos, 0, msg="wrong position, expected: 0, found: {0} ".format(pos)
        )

    def test_sr11_sw4_ch1_set_position_10(self):
        self.audio_source.set_position(10)
        pos = self.audio_source.get_position()
        self.assertEqual(
            pos,
            10,
            msg="wrong position, expected: 10, found: {0} ".format(pos),
        )

    def test_sr11_sw4_ch1_get_time_position_0(self):
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            0.0,
            msg="wrong time position, expected: 0.0, found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_get_time_position_1(self):
        srate = self.audio_source.get_sampling_rate()
        # read one second
        self.audio_source.read(srate)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_get_time_position_0_63(self):
        # read 2.5 seconds
        self.audio_source.read(7)
        tp = self.audio_source.get_time_position()
        self.assertAlmostEqual(
            tp,
            0.636363636364,
            msg="wrong time position, expected: 0.636363636364, "
            "found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_set_time_position_0(self):
        self.audio_source.read(10)
        self.audio_source.set_time_position(0)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            0.0,
            msg="wrong time position, expected: 0.0, found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_set_time_position_1(self):

        self.audio_source.set_time_position(1)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_set_time_position_end(self):
        self.audio_source.set_time_position(100)
        tp = self.audio_source.get_time_position()
        self.assertEqual(
            tp,
            1.0,
            msg="wrong time position, expected: 1.0, found: {0} ".format(tp),
        )

    def test_sr11_sw4_ch1_rewind(self):
        self.audio_source.read(10)
        self.audio_source.rewind()
        tp = self.audio_source.get_position()
        self.assertEqual(
            tp, 0, msg="wrong position, expected: 0.0, found: {0} ".format(tp)
        )

    def test_sr11_sw4_ch1_set_data(self):
        self.audio_source.set_data(b"abcdefgh")
        block = self.audio_source.read(9999)
        exp = b"abcdefgh"
        self.assertEqual(
            block,
            exp,
            msg="wrong block, expected: {}, found: {} ".format(exp, block),
        )

    def test_sr11_sw4_ch1_set_data_exception(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            self.audio_source.set_data(b"abcdef")
        self.assertEqual(
            "The length of audio data must be an integer "
            "multiple of `sample_width * channels`",
            str(audio_param_err.exception),
        )

    def test_sr11_sw4_ch1_append_data_exception(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            self.audio_source.append_data(b"abcdef")
        self.assertEqual(
            "The length of audio data must be an integer "
            "multiple of `sample_width * channels`",
            str(audio_param_err.exception),
        )


class TestBufferAudioSourceCreationException(unittest.TestCase):
    def test_wrong_sample_width_value(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            _ = BufferAudioSource(
                data_buffer=b"ABCDEFGHI",
                sampling_rate=9,
                sample_width=3,
                channels=1,
            )
        self.assertEqual(
            "Sample width must be one of: 1, 2 or 4 (bytes)",
            str(audio_param_err.exception),
        )

    def test_wrong_channels_value(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            _ = BufferAudioSource(
                data_buffer=b"ABCDEFGH",
                sampling_rate=8,
                sample_width=2,
                channels=2,
            )
        self.assertEqual(
            "Only mono audio is currently supported",
            str(audio_param_err.exception),
        )

    def test_wrong_data_buffer_size(self):
        with self.assertRaises(AudioParameterError) as audio_param_err:
            _ = BufferAudioSource(
                data_buffer=b"ABCDEFGHI",
                sampling_rate=8,
                sample_width=2,
                channels=2,
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
