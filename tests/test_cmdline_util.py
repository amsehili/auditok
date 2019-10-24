import os
from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory
from collections import namedtuple
from genty import genty, genty_dataset

from auditok.cmdline_util import (
    _AUDITOK_LOGGER,
    make_kwargs,
    make_duration_formatter,
    make_logger,
    initialize_workers,
    KeywordArguments,
)
from auditok.workers import (
    StreamSaverWorker,
    RegionSaverWorker,
    PlayerWorker,
    CommandLineWorker,
    PrintWorker,
)
from auditok.exceptions import TimeFormatError

_ArgsNamespece = namedtuple(
    "_ArgsNamespece",
    [
        "input",
        "max_read",
        "analysis_window",
        "sampling_rate",
        "sample_width",
        "channels",
        "use_channel",
        "input_format",
        "output_format",
        "large_file",
        "frame_per_buffer",
        "input_device_index",
        "save_stream",
        "save_detections_as",
        "plot",
        "save_image",
        "min_duration",
        "max_duration",
        "max_silence",
        "drop_trailing_silence",
        "strict_min_duration",
        "energy_threshold",
        "echo",
        "progress_bar",
        "command",
        "quiet",
        "printf",
        "time_format",
        "timestamp_format",
    ],
)


@genty
class TestCmdLineUtil(TestCase):
    @genty_dataset(
        no_record=("stream.ogg", None, False, None, "mix", "mix", False),
        no_record_plot=("stream.ogg", None, True, None, None, None, False),
        no_record_save_image=(
            "stream.ogg",
            None,
            True,
            "image.png",
            None,
            None,
            False,
        ),
        record_plot=(None, None, True, None, None, None, True),
        record_save_image=(None, None, False, "image.png", None, None, True),
        int_use_channel=("stream.ogg", None, False, None, "1", 1, False),
        save_detections_as=(
            "stream.ogg",
            "{id}.wav",
            False,
            None,
            None,
            None,
            False,
        ),
    )
    def test_make_kwargs(
        self,
        save_stream,
        save_detections_as,
        plot,
        save_image,
        use_channel,
        exp_use_channel,
        exp_record,
    ):

        args = (
            "file",
            30,
            0.01,
            16000,
            2,
            2,
            use_channel,
            "raw",
            "ogg",
            True,
            None,
            1,
            save_stream,
            save_detections_as,
            plot,
            save_image,
            0.2,
            10,
            0.3,
            False,
            False,
            55,
        )
        misc = (
            False,
            False,
            None,
            True,
            None,
            "TIME_FORMAT",
            "TIMESTAMP_FORMAT",
        )
        args_ns = _ArgsNamespece(*(args + misc))

        io_kwargs = {
            "input": "file",
            "max_read": 30,
            "block_dur": 0.01,
            "sampling_rate": 16000,
            "sample_width": 2,
            "channels": 2,
            "use_channel": exp_use_channel,
            "save_stream": save_stream,
            "save_detections_as": save_detections_as,
            "audio_format": "raw",
            "export_format": "ogg",
            "large_file": True,
            "frames_per_buffer": None,
            "input_device_index": 1,
            "record": exp_record,
        }

        split_kwargs = {
            "min_dur": 0.2,
            "max_dur": 10,
            "max_silence": 0.3,
            "drop_trailing_silence": False,
            "strict_min_dur": False,
            "energy_threshold": 55,
        }

        miscellaneous = {
            "echo": False,
            "command": None,
            "progress_bar": False,
            "quiet": True,
            "printf": None,
            "time_format": "TIME_FORMAT",
            "timestamp_format": "TIMESTAMP_FORMAT",
        }

        expected = KeywordArguments(io_kwargs, split_kwargs, miscellaneous)
        kwargs = make_kwargs(args_ns)
        self.assertEqual(kwargs, expected)

    @genty_dataset(
        only_seconds=("%S", 5400, "5400.000"),
        only_millis=("%I", 5400, "5400000"),
        full=("%h:%m:%s.%i", 3725.365, "01:02:05.365"),
        full_zero_hours=("%h:%m:%s.%i", 1925.075, "00:32:05.075"),
        full_zero_minutes=("%h:%m:%s.%i", 3659.075, "01:00:59.075"),
        full_zero_seconds=("%h:%m:%s.%i", 3720.075, "01:02:00.075"),
        full_zero_millis=("%h:%m:%s.%i", 3725, "01:02:05.000"),
        duplicate_directive=(
            "%h %h:%m:%s.%i %s",
            3725.365,
            "01 01:02:05.365 05",
        ),
        no_millis=("%h:%m:%s", 3725, "01:02:05"),
        no_seconds=("%h:%m", 3725, "01:02"),
        no_minutes=("%h", 3725, "01"),
        no_hours=("%m:%s.%i", 3725, "02:05.000"),
    )
    def test_make_duration_formatter(self, fmt, duration, expected):
        formatter = make_duration_formatter(fmt)
        result = formatter(duration)
        self.assertEqual(result, expected)

    @genty_dataset(
        duplicate_only_seconds=("%S %S",),
        duplicate_only_millis=("%I %I",),
        unknown_directive=("%x",),
    )
    def test_make_duration_formatter_error(self, fmt):
        with self.assertRaises(TimeFormatError):
            make_duration_formatter(fmt)

    def test_make_logger_stderr_and_file(self):
        with TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, "file.log")
            logger = make_logger(stderr=True, file=file)
            self.assertEqual(logger.name, _AUDITOK_LOGGER)
            self.assertEqual(len(logger.handlers), 2)
            self.assertEqual(logger.handlers[0].stream.name, "<stderr>")
            self.assertEqual(logger.handlers[1].stream.name, file)

    def test_make_logger_None(self):
        logger = make_logger(stderr=False, file=None)
        self.assertIsNone(logger)

    def test_initialize_workers_all(self):
        with patch("auditok.cmdline_util.player_for") as patched_player_for:
            with TemporaryDirectory() as tmpdir:
                export_filename = os.path.join(tmpdir, "output.wav")
                reader, observers = initialize_workers(
                    input="tests/data/test_16KHZ_mono_400Hz.wav",
                    save_stream=export_filename,
                    export_format="wave",
                    save_detections_as="{id}.wav",
                    echo=True,
                    progress_bar=False,
                    command="some command",
                    quiet=False,
                    printf="abcd",
                    time_format="%S",
                    timestamp_format="%h:%M:%S",
                )
                reader.stop()
                self.assertTrue(patched_player_for.called)
                self.assertIsInstance(reader, StreamSaverWorker)
                for obs, cls in zip(
                    observers,
                    [
                        RegionSaverWorker,
                        PlayerWorker,
                        CommandLineWorker,
                        PrintWorker,
                    ],
                ):
                    self.assertIsInstance(obs, cls)

    def test_initialize_workers_no_RegionSaverWorker(self):
        with patch("auditok.cmdline_util.player_for") as patched_player_for:
            with TemporaryDirectory() as tmpdir:
                export_filename = os.path.join(tmpdir, "output.wav")
                reader, observers = initialize_workers(
                    input="tests/data/test_16KHZ_mono_400Hz.wav",
                    save_stream=export_filename,
                    export_format="wave",
                    save_detections_as=None,
                    echo=True,
                    progress_bar=False,
                    command="some command",
                    quiet=False,
                    printf="abcd",
                    time_format="%S",
                    timestamp_format="%h:%M:%S",
                )
                reader.stop()
                self.assertTrue(patched_player_for.called)
                self.assertIsInstance(reader, StreamSaverWorker)
                for obs, cls in zip(
                    observers, [PlayerWorker, CommandLineWorker, PrintWorker]
                ):
                    self.assertIsInstance(obs, cls)

    def test_initialize_workers_no_PlayerWorker(self):
        with patch("auditok.cmdline_util.player_for") as patched_player_for:
            with TemporaryDirectory() as tmpdir:
                export_filename = os.path.join(tmpdir, "output.wav")
                reader, observers = initialize_workers(
                    input="tests/data/test_16KHZ_mono_400Hz.wav",
                    save_stream=export_filename,
                    export_format="wave",
                    save_detections_as="{id}.wav",
                    echo=False,
                    progress_bar=False,
                    command="some command",
                    quiet=False,
                    printf="abcd",
                    time_format="%S",
                    timestamp_format="%h:%M:%S",
                )
                reader.stop()
                self.assertFalse(patched_player_for.called)
                self.assertIsInstance(reader, StreamSaverWorker)
                for obs, cls in zip(
                    observers,
                    [RegionSaverWorker, CommandLineWorker, PrintWorker],
                ):
                    self.assertIsInstance(obs, cls)

    def test_initialize_workers_no_CommandLineWorker(self):
        with patch("auditok.cmdline_util.player_for") as patched_player_for:
            with TemporaryDirectory() as tmpdir:
                export_filename = os.path.join(tmpdir, "output.wav")
                reader, observers = initialize_workers(
                    input="tests/data/test_16KHZ_mono_400Hz.wav",
                    save_stream=export_filename,
                    export_format="wave",
                    save_detections_as="{id}.wav",
                    echo=True,
                    progress_bar=False,
                    command=None,
                    quiet=False,
                    printf="abcd",
                    time_format="%S",
                    timestamp_format="%h:%M:%S",
                )
                reader.stop()
                self.assertTrue(patched_player_for.called)
                self.assertIsInstance(reader, StreamSaverWorker)
                for obs, cls in zip(
                    observers, [RegionSaverWorker, PlayerWorker, PrintWorker]
                ):
                    self.assertIsInstance(obs, cls)

    def test_initialize_workers_no_PrintWorker(self):
        with patch("auditok.cmdline_util.player_for") as patched_player_for:
            with TemporaryDirectory() as tmpdir:
                export_filename = os.path.join(tmpdir, "output.wav")
                reader, observers = initialize_workers(
                    input="tests/data/test_16KHZ_mono_400Hz.wav",
                    save_stream=export_filename,
                    export_format="wave",
                    save_detections_as="{id}.wav",
                    echo=True,
                    progress_bar=False,
                    command="some command",
                    quiet=True,
                    printf="abcd",
                    time_format="%S",
                    timestamp_format="%h:%M:%S",
                )
                reader.stop()
                self.assertTrue(patched_player_for.called)
                self.assertIsInstance(reader, StreamSaverWorker)
                for obs, cls in zip(
                    observers,
                    [RegionSaverWorker, PlayerWorker, CommandLineWorker],
                ):
                    self.assertIsInstance(obs, cls)

    def test_initialize_workers_no_observers(self):
        with patch("auditok.cmdline_util.player_for") as patched_player_for:
            reader, observers = initialize_workers(
                input="tests/data/test_16KHZ_mono_400Hz.wav",
                save_stream=None,
                export_format="wave",
                save_detections_as=None,
                echo=True,
                progress_bar=False,
                command=None,
                quiet=True,
                printf="abcd",
                time_format="%S",
                timestamp_format="%h:%M:%S",
            )
            self.assertTrue(patched_player_for.called)
            self.assertFalse(isinstance(reader, StreamSaverWorker))
            self.assertTrue(len(observers), 0)
