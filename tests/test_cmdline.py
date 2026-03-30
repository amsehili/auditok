"""Integration tests for CLI subcommand dispatch, trim, and fix-pauses."""

import os
from tempfile import TemporaryDirectory

import pytest

from auditok.cmdline import main

WAV_FILE = "tests/data/test_16KHZ_mono_400Hz.wav"


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


class TestDeprecationWarnings:
    def test_join_detections_warns(self, capsys):
        with TemporaryDirectory() as tmpdir:
            stream = os.path.join(tmpdir, "stream.wav")
            ret = main(["split", WAV_FILE, "-q", "-j", "0.5", "-O", stream])
            assert ret == 0
            captured = capsys.readouterr()
            assert "--join-detections" in captured.err
            assert "deprecated" in captured.err.lower()
