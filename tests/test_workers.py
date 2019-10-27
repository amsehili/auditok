import os
import unittest
from unittest import TestCase
from unittest.mock import patch, call, Mock
from tempfile import TemporaryDirectory
from genty import genty, genty_dataset
from auditok import AudioRegion, AudioDataSource
from auditok.exceptions import AudioEncodingWarning
from auditok.cmdline_util import make_logger
from auditok.workers import (
    TokenizerWorker,
    StreamSaverWorker,
    RegionSaverWorker,
    PlayerWorker,
    CommandLineWorker,
    PrintWorker,
)


@genty
class TestWorkers(TestCase):
    def setUp(self):

        self.reader = AudioDataSource(
            input="tests/data/test_split_10HZ_mono.raw",
            block_dur=0.1,
            sr=10,
            sw=2,
            ch=1,
        )
        self.expected = [
            (0.2, 1.6),
            (1.7, 3.1),
            (3.4, 5.4),
            (5.4, 7.4),
            (7.4, 7.6),
        ]

    def tearDown(self):
        self.reader.close()

    def test_TokenizerWorker(self):
        with TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, "file.log")
            logger = make_logger(file=file, name="test_TokenizerWorker")
            tokenizer = TokenizerWorker(
                self.reader,
                logger=logger,
                min_dur=0.3,
                max_dur=2,
                max_silence=0.2,
                drop_trailing_silence=False,
                strict_min_dur=False,
                eth=50,
            )
            tokenizer.start_all()
            tokenizer.join()
            # Get logged text
            with open(file) as fp:
                log_lines = fp.readlines()

        log_fmt = "[DET]: Detection {} (start: {:.3f}, "
        log_fmt += "end: {:.3f}, duration: {:.3f})"
        self.assertEqual(len(tokenizer.detections), len(self.expected))
        for i, (det, exp, log_line) in enumerate(
            zip(tokenizer.detections, self.expected, log_lines), 1
        ):
            start, end = exp
            exp_log_line = log_fmt.format(i, start, end, end - start)
            self.assertAlmostEqual(det.start, start)
            self.assertAlmostEqual(det.end, end)
            # remove timestamp part and strip new line
            self.assertEqual(log_line[28:].strip(), exp_log_line)

    def test_PlayerWorker(self):
        with TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, "file.log")
            logger = make_logger(file=file, name="test_RegionSaverWorker")
            player_mock = Mock()
            observers = [PlayerWorker(player_mock, logger=logger)]
            tokenizer = TokenizerWorker(
                self.reader,
                logger=logger,
                observers=observers,
                min_dur=0.3,
                max_dur=2,
                max_silence=0.2,
                drop_trailing_silence=False,
                strict_min_dur=False,
                eth=50,
            )
            tokenizer.start_all()
            tokenizer.join()
            tokenizer._observers[0].join()
            # Get logged text
            with open(file) as fp:
                log_lines = [
                    line
                    for line in fp.readlines()
                    if line.startswith("[PLAY]")
                ]
        self.assertTrue(player_mock.play.called)

        self.assertEqual(len(tokenizer.detections), len(self.expected))
        log_fmt = "[PLAY]: Detection {id} played"
        for i, (det, exp, log_line) in enumerate(
            zip(tokenizer.detections, self.expected, log_lines), 1
        ):
            start, end = exp
            exp_log_line = log_fmt.format(id=i)
            self.assertAlmostEqual(det.start, start)
            self.assertAlmostEqual(det.end, end)
            # Remove timestamp part and strip new line
            self.assertEqual(log_line[28:].strip(), exp_log_line)

    def test_RegionSaverWorker(self):
        filename_format = (
            "Region_{id}_{start:.6f}-{end:.3f}_{duration:.3f}.wav"
        )
        with TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, "file.log")
            logger = make_logger(file=file, name="test_RegionSaverWorker")
            observers = [RegionSaverWorker(filename_format, logger=logger)]
            tokenizer = TokenizerWorker(
                self.reader,
                logger=logger,
                observers=observers,
                min_dur=0.3,
                max_dur=2,
                max_silence=0.2,
                drop_trailing_silence=False,
                strict_min_dur=False,
                eth=50,
            )
            with patch("auditok.core.AudioRegion.save") as patched_save:
                tokenizer.start_all()
                tokenizer.join()
                tokenizer._observers[0].join()
            # Get logged text
            with open(file) as fp:
                log_lines = [
                    line
                    for line in fp.readlines()
                    if line.startswith("[SAVE]")
                ]

        # Assert RegionSaverWorker ran as expected
        expected_save_calls = [
            call(
                filename_format.format(
                    id=i, start=exp[0], end=exp[1], duration=exp[1] - exp[0]
                ),
                None,
            )
            for i, exp in enumerate(self.expected, 1)
        ]

        # Get calls to 'AudioRegion.save'
        mock_calls = [
            c for i, c in enumerate(patched_save.mock_calls) if i % 2 == 0
        ]
        self.assertEqual(mock_calls, expected_save_calls)
        self.assertEqual(len(tokenizer.detections), len(self.expected))

        log_fmt = "[SAVE]: Detection {id} saved as '{filename}'"
        for i, (det, exp, log_line) in enumerate(
            zip(tokenizer.detections, self.expected, log_lines), 1
        ):
            start, end = exp
            expected_filename = filename_format.format(
                id=i, start=start, end=end, duration=end - start
            )
            exp_log_line = log_fmt.format(i, expected_filename)
            self.assertAlmostEqual(det.start, start)
            self.assertAlmostEqual(det.end, end)
            # Remove timestamp part and strip new line
            self.assertEqual(log_line[28:].strip(), exp_log_line)

    def test_CommandLineWorker(self):
        command_format = "do nothing with"
        with TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, "file.log")
            logger = make_logger(file=file, name="test_CommandLineWorker")
            observers = [CommandLineWorker(command_format, logger=logger)]
            tokenizer = TokenizerWorker(
                self.reader,
                logger=logger,
                observers=observers,
                min_dur=0.3,
                max_dur=2,
                max_silence=0.2,
                drop_trailing_silence=False,
                strict_min_dur=False,
                eth=50,
            )
            with patch("auditok.workers.os.system") as patched_os_system:
                tokenizer.start_all()
                tokenizer.join()
                tokenizer._observers[0].join()
            # Get logged text
            with open(file) as fp:
                log_lines = [
                    line
                    for line in fp.readlines()
                    if line.startswith("[COMMAND]")
                ]

        # Assert CommandLineWorker ran as expected
        expected_save_calls = [call(command_format) for _ in self.expected]
        self.assertEqual(patched_os_system.mock_calls, expected_save_calls)
        self.assertEqual(len(tokenizer.detections), len(self.expected))
        log_fmt = "[COMMAND]: Detection {id} command '{command}'"
        for i, (det, exp, log_line) in enumerate(
            zip(tokenizer.detections, self.expected, log_lines), 1
        ):
            start, end = exp
            exp_log_line = log_fmt.format(i, command_format)
            self.assertAlmostEqual(det.start, start)
            self.assertAlmostEqual(det.end, end)
            # Remove timestamp part and strip new line
            self.assertEqual(log_line[28:].strip(), exp_log_line)

    def test_PrintWorker(self):
        observers = [
            PrintWorker(print_format="[{id}] {start} {end}, dur: {duration}")
        ]
        tokenizer = TokenizerWorker(
            self.reader,
            observers=observers,
            min_dur=0.3,
            max_dur=2,
            max_silence=0.2,
            drop_trailing_silence=False,
            strict_min_dur=False,
            eth=50,
        )
        with patch("builtins.print") as patched_print:
            tokenizer.start_all()
            tokenizer.join()
            tokenizer._observers[0].join()

        # Assert PrintWorker ran as expected
        expected_print_calls = [
            call(
                "[{}] {:.3f} {:.3f}, dur: {:.3f}".format(
                    i, *exp, exp[1] - exp[0]
                )
            )
            for i, exp in enumerate(self.expected, 1)
        ]
        self.assertEqual(patched_print.mock_calls, expected_print_calls)
        self.assertEqual(len(tokenizer.detections), len(self.expected))
        for det, exp in zip(tokenizer.detections, self.expected):
            start, end = exp
            self.assertAlmostEqual(det.start, start)
            self.assertAlmostEqual(det.end, end)

    def test_StreamSaverWorker_wav(self):
        with TemporaryDirectory() as tmpdir:
            expected_filename = os.path.join(tmpdir, "output.wav")
            saver = StreamSaverWorker(self.reader, expected_filename)
            saver.start()

            tokenizer = TokenizerWorker(saver)
            tokenizer.start_all()
            tokenizer.join()
            saver.join()

            output_filename = saver.save_stream()
            region = AudioRegion.load(
                "tests/data/test_split_10HZ_mono.raw", sr=10, sw=2, ch=1
            )

            expected_region = AudioRegion.load(output_filename)
            self.assertEqual(output_filename, expected_filename)
            self.assertEqual(region, expected_region)
            self.assertEqual(saver.data, bytes(expected_region))

    def test_StreamSaverWorker_raw(self):
        with TemporaryDirectory() as tmpdir:
            expected_filename = os.path.join(tmpdir, "output")
            saver = StreamSaverWorker(
                self.reader, expected_filename, export_format="raw"
            )
            saver.start()
            tokenizer = TokenizerWorker(saver)
            tokenizer.start_all()
            tokenizer.join()
            saver.join()
            output_filename = saver.save_stream()
            region = AudioRegion.load(
                "tests/data/test_split_10HZ_mono.raw", sr=10, sw=2, ch=1
            )
            expected_region = AudioRegion.load(
                output_filename, sr=10, sw=2, ch=1, audio_format="raw"
            )
            self.assertEqual(output_filename, expected_filename)
            self.assertEqual(region, expected_region)
            self.assertEqual(saver.data, bytes(expected_region))

    def test_StreamSaverWorker_encode_audio(self):
        with TemporaryDirectory() as tmpdir:
            with patch("auditok.workers._run_subprocess") as patch_rsp:
                patch_rsp.return_value = (1, None, None)
                expected_filename = os.path.join(tmpdir, "output.ogg")
                tmp_expected_filename = expected_filename + ".wav"
                saver = StreamSaverWorker(self.reader, expected_filename)
                saver.start()
                tokenizer = TokenizerWorker(saver)
                tokenizer.start_all()
                tokenizer.join()
                saver.join()
                with self.assertRaises(AudioEncodingWarning) as rt_warn:
                    saver.save_stream()
            warn_msg = "Couldn't save audio data in the desired format "
            warn_msg += "'ogg'. Either none of 'ffmpeg', 'avconv' or 'sox' "
            warn_msg += "is installed or this format is not recognized.\n"
            warn_msg += "Audio file was saved as '{}'"
            self.assertEqual(
                warn_msg.format(tmp_expected_filename), str(rt_warn.exception)
            )
            ffmpef_avconv = [
                "-y",
                "-f",
                "wav",
                "-i",
                tmp_expected_filename,
                "-f",
                "ogg",
                expected_filename,
            ]
            expected_calls = [
                call(["ffmpeg"] + ffmpef_avconv),
                call(["avconv"] + ffmpef_avconv),
                call(
                    [
                        "sox",
                        "-t",
                        "wav",
                        tmp_expected_filename,
                        expected_filename,
                    ]
                ),
            ]
            self.assertEqual(patch_rsp.mock_calls, expected_calls)
            region = AudioRegion.load(
                "tests/data/test_split_10HZ_mono.raw", sr=10, sw=2, ch=1
            )
            self.assertTrue(saver._exported)
            self.assertEqual(saver.data, bytes(region))


if __name__ == "__main__":
    unittest.main()
