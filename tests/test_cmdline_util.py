from unittest import TestCase
from collections import namedtuple
from genty import genty, genty_dataset

from auditok.cmdline_util import (
    LOGGER_NAME,
    make_kwargs,
    make_duration_fromatter,
    make_logger,
    initialize_workers,
    KeywordArguments,
)
from auditok.workers import StreamSaverWorker
from auditok.exceptions import TimeFormatError

_ArgsNamespece = namedtuple(
    "_ArgsNamespece",
    [
        "input",
        "max_time",
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
        "plot",
        "save_image",
        "min_duration",
        "max_duration",
        "max_silence",
        "drop_trailing_silence",
        "strict_min_duration",
        "energy_threshold",
        "echo",
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
        no_record=("stream.ogg", False, None, "mix", "mix", False),
        no_record_plot=("stream.ogg", True, None, None, None, False),
        no_record_save_image=(
            "stream.ogg",
            True,
            "image.png",
            None,
            None,
            False,
        ),
        record_plot=(None, True, None, None, None, True),
        record_save_image=(None, False, "image.png", None, None, True),
        int_use_channel=("stream.ogg", False, None, "1", 1, False),
    )
    def test_make_kwargs(
        self,
        save_stream,
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
            plot,
            save_image,
            0.2,
            10,
            0.3,
            False,
            False,
            55,
        )
        misc = (False, None, True, None, "TIME_FORMAT", "TIMESTAMP_FORMAT")

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
            "quiet": True,
            "printf": None,
            "time_format": "TIME_FORMAT",
            "timestamp_format": "TIMESTAMP_FORMAT",
        }

        expected = KeywordArguments(io_kwargs, split_kwargs, miscellaneous)
        kwargs = make_kwargs(args_ns)
        self.assertEqual(kwargs, expected)
