import os
import pytest
from tempfile import TemporaryDirectory
from collections import namedtuple
from unittest.mock import patch

from auditok.cmdline_util import (
    _AUDITOK_LOGGER,
    make_kwargs,
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

_ArgsNamespace = namedtuple(
    "_ArgsNamespace",
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


@pytest.mark.parametrize(
    "save_stream, save_detections_as, plot, save_image, use_channel, exp_use_channel, exp_record",
    [
        # no_record
        ("stream.ogg", None, False, None, "mix", "mix", False),
        # no_record_plot
        ("stream.ogg", None, True, None, None, None, False),
        # no_record_save_image
        ("stream.ogg", None, True, "image.png", None, None, False),
        # record_plot
        (None, None, True, None, None, None, True),
        # record_save_image
        (None, None, False, "image.png", None, None, True),
        # int_use_channel
        ("stream.ogg", None, False, None, "1", 1, False),
        # save_detections_as
        ("stream.ogg", "{id}.wav", False, None, None, None, False),
    ],
    ids=[
        "no_record",
        "no_record_plot",
        "no_record_save_image",
        "record_plot",
        "record_save_image",
        "int_use_channel",
        "save_detections_as",
    ],
)
def test_make_kwargs(
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
    args_ns = _ArgsNamespace(*(args + misc))

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
    assert kwargs == expected


def test_make_logger_stderr_and_file(capsys):
    with TemporaryDirectory() as tmpdir:
        file = os.path.join(tmpdir, "file.log")
        logger = make_logger(stderr=True, file=file)
        assert logger.name == _AUDITOK_LOGGER
        assert len(logger.handlers) == 2
        assert logger.handlers[1].stream.name == file
        logger.info("This is a debug message")
        captured = capsys.readouterr()
        assert "This is a debug message" in captured.err


def test_make_logger_None():
    logger = make_logger(stderr=False, file=None)
    assert logger is None


def test_initialize_workers_all():
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
            assert patched_player_for.called
            assert isinstance(reader, StreamSaverWorker)
            for obs, cls in zip(
                observers,
                [
                    RegionSaverWorker,
                    PlayerWorker,
                    CommandLineWorker,
                    PrintWorker,
                ],
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_no_RegionSaverWorker():
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
            assert patched_player_for.called
            assert isinstance(reader, StreamSaverWorker)
            for obs, cls in zip(
                observers, [PlayerWorker, CommandLineWorker, PrintWorker]
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_no_PlayerWorker():
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
            assert not patched_player_for.called
            assert isinstance(reader, StreamSaverWorker)
            for obs, cls in zip(
                observers,
                [RegionSaverWorker, CommandLineWorker, PrintWorker],
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_no_CommandLineWorker():
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
            assert patched_player_for.called
            assert isinstance(reader, StreamSaverWorker)
            for obs, cls in zip(
                observers, [RegionSaverWorker, PlayerWorker, PrintWorker]
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_no_PrintWorker():
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
            assert patched_player_for.called
            assert isinstance(reader, StreamSaverWorker)
            for obs, cls in zip(
                observers,
                [RegionSaverWorker, PlayerWorker, CommandLineWorker],
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_no_observers():
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
        assert patched_player_for.called
        assert not isinstance(reader, StreamSaverWorker)
        assert len(observers) == 1
