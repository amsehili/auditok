import os
from unittest import TestCase
from unittest.mock import patch, call
from tempfile import TemporaryDirectory
from genty import genty, genty_dataset
from auditok import AudioDataSource
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

        # Asser PrintWorker ran as expected
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
