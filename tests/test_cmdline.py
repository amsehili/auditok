"""Tests for CLI subcommand dispatch, kwargs builders, and worker setup."""

import os
from collections import namedtuple
from shutil import which
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from auditok.cmdline import (
    _AUDITOK_LOGGER,
    _build_split_command_kwargs,
    _SplitCommandKwargs,
    initialize_workers,
    main,
    make_logger,
)
from auditok.exceptions import ArgumentError
from auditok.workers import (
    AudioEventsJoinerWorker,
    CommandLineWorker,
    PlayerWorker,
    PrintWorker,
    RegionSaverWorker,
    StreamSaverWorker,
)

requires_ffmpeg = pytest.mark.skipif(
    which("ffmpeg") is None, reason="ffmpeg not available"
)

WAV_FILE = "tests/data/test_16KHZ_mono_400Hz.wav"
MP3_FILE_STEREO_44K = "tests/data/DTMF_tones_44.1KHZ_stereo.mp3"


# ── Subcommand dispatch ────────────────────────────────────────────


class TestSubcommandDispatch:
    def test_version(self, capsys):
        ret = main(["--version"])
        assert ret == 0
        assert "0.4.0" in capsys.readouterr().out

    def test_version_short(self, capsys):
        ret = main(["-v"])
        assert ret == 0
        assert "0.4.0" in capsys.readouterr().out

    def test_help_shows_subcommands(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "split" in out
        assert "trim" in out
        assert "fix-pauses" in out

    def test_split_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["split", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "auditok split" in out

    def test_trim_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["trim", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "auditok trim" in out

    def test_fix_pauses_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["fix-pauses", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "auditok fix-pauses" in out

    def test_implicit_split_with_file(self, capsys):
        """auditok file.wav ... should behave like auditok split file.wav ..."""
        ret = main([WAV_FILE, "-q"])
        assert ret == 0

    def test_explicit_split_with_file(self, capsys):
        ret = main(["split", WAV_FILE, "-q"])
        assert ret == 0


# ── trim subcommand ────────────────────────────────────────────────


class TestTrimSubcommand:
    def test_trim_file_creates_output(self):
        with TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "trimmed.wav")
            ret = main(["trim", WAV_FILE, "-o", output])
            assert ret == 0
            assert os.path.exists(output)
            assert os.path.getsize(output) > 0

    def test_trim_no_activity(self, capsys):
        with TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "trimmed.wav")
            ret = main(["trim", WAV_FILE, "-o", output, "-e", "99"])
            assert ret == 0
            captured = capsys.readouterr()
            assert "No audio activity" in captured.err

    def test_trim_with_output_format(self):
        with TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "trimmed.raw")
            ret = main(["trim", WAV_FILE, "-o", output, "-T", "raw"])
            assert ret == 0
            assert os.path.exists(output)
            assert os.path.getsize(output) > 0

    def test_trim_requires_output(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["trim", WAV_FILE])
        assert exc_info.value.code != 0


# ── fix-pauses subcommand ──────────────────────────────────────────


class TestFixPausesSubcommand:
    def test_fix_pauses_file_creates_output(self):
        with TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "fixed.wav")
            ret = main(["fix-pauses", WAV_FILE, "-o", output, "-d", "0.5"])
            assert ret == 0
            assert os.path.exists(output)
            assert os.path.getsize(output) > 0

    def test_fix_pauses_no_activity(self, capsys):
        with TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "fixed.wav")
            ret = main(
                ["fix-pauses", WAV_FILE, "-o", output, "-d", "0.5", "-e", "99"]
            )
            assert ret == 0
            captured = capsys.readouterr()
            assert "No audio activity" in captured.err

    def test_fix_pauses_requires_output(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["fix-pauses", WAV_FILE, "-d", "0.5"])
        assert exc_info.value.code != 0

    def test_fix_pauses_requires_pause_duration(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["fix-pauses", WAV_FILE, "-o", "out.wav"])
        assert exc_info.value.code != 0


# ── Deprecation warnings ──────────────────────────────────────────


# ── Audio params not forced on file input ─────────────────────────


@requires_ffmpeg
class TestAudioParamsNotForcedOnFileInput:
    """CLI must not pass default -r/-w/-c to FFmpegAudioSource so that
    the original audio format is preserved."""

    def _capture_ffmpeg_init_args(self, cli_args):
        """Run main() and return the (sr, sw, ch) passed to
        FFmpegAudioSource.__init__."""
        captured = {}
        original_init = __import__(
            "auditok.io", fromlist=["FFmpegAudioSource"]
        ).FFmpegAudioSource.__init__

        def spy_init(
            self_,
            filename,
            sampling_rate=None,
            sample_width=None,
            channels=None,
        ):
            captured["sr"] = sampling_rate
            captured["sw"] = sample_width
            captured["ch"] = channels
            return original_init(
                self_,
                filename,
                sampling_rate=sampling_rate,
                sample_width=sample_width,
                channels=channels,
            )

        with patch("auditok.io.FFmpegAudioSource.__init__", spy_init):
            main(cli_args)
        return captured

    def test_split_mp3_no_audio_params(self):
        """split with mp3 and no -r/-w/-c: FFmpegAudioSource must get
        None for all three params."""
        cap = self._capture_ffmpeg_init_args(
            ["split", MP3_FILE_STEREO_44K, "-q"]
        )
        assert cap["sr"] is None
        assert cap["sw"] is None
        assert cap["ch"] is None

    def test_split_mp3_explicit_rate_ignored(self):
        """split with explicit -r on a file: ignored, original format
        is preserved."""
        cap = self._capture_ffmpeg_init_args(
            ["split", MP3_FILE_STEREO_44K, "-q", "-r", "8000"]
        )
        assert cap["sr"] is None
        assert cap["sw"] is None
        assert cap["ch"] is None

    def test_trim_mp3_no_audio_params(self):
        """trim with mp3 and no -r/-w/-c: FFmpegAudioSource must get
        None for all three params."""
        with TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "out.wav")
            cap = self._capture_ffmpeg_init_args(
                ["trim", MP3_FILE_STEREO_44K, "-o", output]
            )
        assert cap["sr"] is None
        assert cap["sw"] is None
        assert cap["ch"] is None

    def test_fix_pauses_mp3_no_audio_params(self):
        """fix-pauses with mp3 and no -r/-w/-c: FFmpegAudioSource must
        get None for all three params."""
        with TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "out.wav")
            cap = self._capture_ffmpeg_init_args(
                ["fix-pauses", MP3_FILE_STEREO_44K, "-o", output, "-d", "0.5"]
            )
        assert cap["sr"] is None
        assert cap["sw"] is None
        assert cap["ch"] is None

    def test_split_mp3_preserves_original_format(self):
        """Verify the loaded audio actually has the file's original
        sample rate and channel count (44100 Hz, stereo)."""
        from auditok.io import from_file

        source = from_file(MP3_FILE_STEREO_44K)
        assert source.sr == 44100
        assert source.ch == 2
        source.close()


# ── Deprecation warnings ──────────────────────────────────────────


class TestDeprecationWarnings:
    def test_join_detections_warns(self, capsys):
        with TemporaryDirectory() as tmpdir:
            stream = os.path.join(tmpdir, "stream.wav")
            ret = main(["split", WAV_FILE, "-q", "-j", "0.5", "-O", stream])
            assert ret == 0
            captured = capsys.readouterr()
            assert "--join-detections" in captured.err
            assert "deprecated" in captured.err.lower()


class TestDuplicateOptions:
    """Reject CLI arguments that appear more than once."""

    def test_duplicate_short_flag(self):
        with pytest.raises(SystemExit, match="2"):
            main(["split", "-e", "50", "-e", "65", WAV_FILE])

    def test_duplicate_long_flag(self):
        with pytest.raises(SystemExit, match="2"):
            main(
                [
                    "split",
                    "--energy-threshold",
                    "50",
                    "--energy-threshold",
                    "65",
                    WAV_FILE,
                ]
            )

    def test_short_and_long_alias_conflict(self):
        with pytest.raises(SystemExit, match="2"):
            main(["split", "-e", "50", "--energy-threshold", "65", WAV_FILE])

    def test_duplicate_on_fix_pauses(self):
        with pytest.raises(SystemExit, match="2"):
            main(
                [
                    "fix-pauses",
                    "-e",
                    "50",
                    "-e",
                    "65",
                    "-o",
                    "out.wav",
                    "-d",
                    "1",
                    WAV_FILE,
                ]
            )

    def test_duplicate_on_trim(self):
        with pytest.raises(SystemExit, match="2"):
            main(["trim", "-e", "50", "-e", "65", "-o", "out.wav", WAV_FILE])

    def test_duplicate_with_implicit_split(self):
        with pytest.raises(SystemExit, match="2"):
            main(["-e", "50", "-e", "65", WAV_FILE])

    def test_no_error_when_no_duplicates(self):
        ret = main(["split", WAV_FILE, "-q", "-e", "50"])
        assert ret == 0


# ── _build_split_command_kwargs ───────────────────────────────────

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
        "join_detections",
        "plot",
        "save_image",
        "min_duration",
        "max_duration",
        "max_silence",
        "max_leading_silence",
        "max_trailing_silence",
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
    "save_stream, save_detections_as, join_detections, plot, save_image, use_channel, exp_use_channel, exp_record",  # noqa: B950
    [
        # no_record_no_join
        ("stream.ogg", None, None, False, None, "mix", "mix", False),
        # no_record_plot_join
        ("stream.ogg", None, 1.0, True, None, None, None, False),
        # no_record_save_image
        ("stream.ogg", None, None, True, "image.png", None, None, False),
        # record_plot
        (None, None, None, True, None, None, None, True),
        # record_save_image
        (None, None, None, False, "image.png", None, None, True),
        # int_use_channel
        ("stream.ogg", None, None, False, None, "1", 1, False),
        # save_detections_as
        ("stream.ogg", "{id}.wav", None, False, None, None, None, False),
    ],
    ids=[
        "no_record_no_join",
        "no_record_plot",
        "no_record_save_image",
        "record_plot",
        "record_save_image",
        "int_use_channel",
        "save_detections_as",
    ],
)
def test_build_split_command_kwargs(
    save_stream,
    save_detections_as,
    join_detections,
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
        join_detections,
        plot,
        save_image,
        0.2,
        10,
        0.3,
        0,  # max_leading_silence
        None,  # max_trailing_silence
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
        "join_detections": join_detections,
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
        "max_leading_silence": 0,
        "max_trailing_silence": None,
        "drop_trailing_silence": False,
        "strict_min_dur": False,
        "energy_threshold": 55,
        "analysis_window": 0.01,
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

    expected = _SplitCommandKwargs(io_kwargs, split_kwargs, miscellaneous)
    kwargs = _build_split_command_kwargs(args_ns)
    assert kwargs == expected


def test_build_split_command_kwargs_error():
    args = (
        "file",
        30,
        0.01,
        16000,
        2,
        2,
        1,
        "raw",
        "ogg",
        True,
        None,
        1,
        None,  # save_stream
        None,
        1.0,  # join_detections
        None,
        None,
        0.2,
        10,
        0.3,
        0,  # max_leading_silence
        None,  # max_trailing_silence
        False,
        False,
        55,
        False,
        False,
        None,
        True,
        None,
        "TIME_FORMAT",
        "TIMESTAMP_FORMAT",
    )

    args_ns = _ArgsNamespace(*args)
    expected_err_msg = "using --join-detections/-j requires "
    expected_err_msg += "--save-stream/-O to be specified."
    with pytest.raises(ArgumentError) as arg_err:
        _build_split_command_kwargs(args_ns)
    assert str(arg_err.value) == expected_err_msg


# ── make_logger ──────────────────────────────────────────────────


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


# ── initialize_workers ───────────────────────────────────────────


def test_initialize_workers_all_plus_full_stream_saver():
    with patch("auditok.cmdline.player_for") as patched_player_for:
        with TemporaryDirectory() as tmpdir:
            export_filename = os.path.join(tmpdir, "output.wav")
            reader, tokenizer_worker = initialize_workers(
                input="tests/data/test_16KHZ_mono_400Hz.wav",
                save_stream=export_filename,
                export_format="wave",
                save_detections_as="{id}.wav",
                join_detections=None,
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
                tokenizer_worker._observers,
                [
                    RegionSaverWorker,
                    PlayerWorker,
                    CommandLineWorker,
                    PrintWorker,
                ],
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_all_plus_audio_event_joiner():
    with patch("auditok.cmdline.player_for") as patched_player_for:
        with TemporaryDirectory() as tmpdir:
            export_filename = os.path.join(tmpdir, "output.wav")
            reader, tokenizer_worker = initialize_workers(
                input="tests/data/test_16KHZ_mono_400Hz.wav",
                save_stream=export_filename,
                export_format="wave",
                save_detections_as="{id}.wav",
                join_detections=1,
                echo=True,
                progress_bar=False,
                command="some command",
                quiet=False,
                printf="abcd",
                time_format="%S",
                timestamp_format="%h:%M:%S",
            )
            assert patched_player_for.called
            assert not isinstance(reader, StreamSaverWorker)
            for obs, cls in zip(
                tokenizer_worker._observers,
                [
                    AudioEventsJoinerWorker,
                    RegionSaverWorker,
                    PlayerWorker,
                    CommandLineWorker,
                    PrintWorker,
                ],
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_no_RegionSaverWorker():
    with patch("auditok.cmdline.player_for") as patched_player_for:
        with TemporaryDirectory() as tmpdir:
            export_filename = os.path.join(tmpdir, "output.wav")
            reader, tokenizer_worker = initialize_workers(
                input="tests/data/test_16KHZ_mono_400Hz.wav",
                save_stream=export_filename,
                export_format="wave",
                save_detections_as=None,
                join_detections=None,
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
                tokenizer_worker._observers,
                [PlayerWorker, CommandLineWorker, PrintWorker],
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_no_PlayerWorker():
    with patch("auditok.cmdline.player_for") as patched_player_for:
        with TemporaryDirectory() as tmpdir:
            export_filename = os.path.join(tmpdir, "output.wav")
            reader, tokenizer_worker = initialize_workers(
                input="tests/data/test_16KHZ_mono_400Hz.wav",
                save_stream=export_filename,
                export_format="wave",
                save_detections_as="{id}.wav",
                join_detections=None,
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
                tokenizer_worker._observers,
                [RegionSaverWorker, CommandLineWorker, PrintWorker],
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_no_CommandLineWorker():
    with patch("auditok.cmdline.player_for") as patched_player_for:
        with TemporaryDirectory() as tmpdir:
            export_filename = os.path.join(tmpdir, "output.wav")
            reader, tokenizer_worker = initialize_workers(
                input="tests/data/test_16KHZ_mono_400Hz.wav",
                save_stream=export_filename,
                export_format="wave",
                save_detections_as="{id}.wav",
                join_detections=None,
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
                tokenizer_worker._observers,
                [RegionSaverWorker, PlayerWorker, PrintWorker],
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_no_PrintWorker():
    with patch("auditok.cmdline.player_for") as patched_player_for:
        with TemporaryDirectory() as tmpdir:
            export_filename = os.path.join(tmpdir, "output.wav")
            reader, tokenizer_worker = initialize_workers(
                input="tests/data/test_16KHZ_mono_400Hz.wav",
                save_stream=export_filename,
                export_format="wave",
                save_detections_as="{id}.wav",
                join_detections=None,
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
                tokenizer_worker._observers,
                [RegionSaverWorker, PlayerWorker, CommandLineWorker],
            ):
                assert isinstance(obs, cls)


def test_initialize_workers_no_observers():
    with patch("auditok.cmdline.player_for") as patched_player_for:
        reader, tokenizer_worker = initialize_workers(
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
        assert len(tokenizer_worker._observers) == 1
