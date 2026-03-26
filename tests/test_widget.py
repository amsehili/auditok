import base64
import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from auditok.core import AudioRegion
from auditok.widget import (
    _audio_to_wav_b64,
    _downsample_peaks,
    _in_notebook,
    display_interactive,
)


def test_in_notebook_no_ipython():
    """_in_notebook returns False when IPython is not available."""
    assert _in_notebook() is False


def test_in_notebook_terminal():
    """_in_notebook returns False for TerminalInteractiveShell."""
    mock_shell = MagicMock()
    mock_shell.__class__.__name__ = "TerminalInteractiveShell"
    with patch("builtins.get_ipython", return_value=mock_shell, create=True):
        assert _in_notebook() is False


def test_in_notebook_zmq():
    """_in_notebook returns True for ZMQInteractiveShell."""
    mock_shell = MagicMock()
    mock_shell.__class__.__name__ = "ZMQInteractiveShell"
    with patch("builtins.get_ipython", return_value=mock_shell, create=True):
        assert _in_notebook() is True


def test_audio_to_wav_b64():
    """_audio_to_wav_b64 produces a valid base64-encoded WAV."""
    data = b"\x00\x01" * 80
    b64 = _audio_to_wav_b64(data, sr=16000, sw=2, ch=1)
    wav_bytes = base64.b64decode(b64)
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnchannels() == 1
        assert wf.readframes(80) == data


def test_downsample_peaks_passthrough():
    """When n_bins >= n_samples, peaks are per-sample."""
    samples = np.array([0.0, 0.5, -0.3, 1.0])
    peaks = _downsample_peaks(samples, n_bins=10)
    assert len(peaks) == 4
    for i, s in enumerate(samples):
        assert peaks[i] == [float(s), float(s)]


def test_downsample_peaks_reduction():
    """Downsampling reduces to the requested number of bins."""
    samples = np.random.randn(10000)
    peaks = _downsample_peaks(samples, n_bins=200)
    assert len(peaks) == 200
    for mn, mx in peaks:
        assert mn <= mx


def _capture_widget_html(region, events):
    """Helper: call display_interactive and return the generated HTML string."""
    captured = []
    mock_display = MagicMock(side_effect=lambda obj: captured.append(obj))
    mock_html_cls = MagicMock(side_effect=lambda s: s)
    mock_mod = MagicMock()
    mock_mod.display = mock_display
    mock_mod.HTML = mock_html_cls
    with patch.dict("sys.modules", {"IPython.display": mock_mod}):
        display_interactive(region, events)
    assert len(captured) == 1
    return captured[0]


@pytest.fixture
def widget_html():
    """Return the HTML produced by display_interactive for a short signal."""
    data = b"\x00\x01" * 800  # 0.05s at 16kHz mono 16-bit
    region = AudioRegion(data, 16000, 2, 1)
    events = [
        AudioRegion(data[:400], 16000, 2, 1, start=0.0),
        AudioRegion(data[400:], 16000, 2, 1, start=0.025),
    ]
    return _capture_widget_html(region, events)


def test_display_interactive_html_content(widget_html):
    """display_interactive produces HTML with expected elements."""
    assert "<canvas" in widget_html
    assert "auditok" in widget_html
    assert "Play all" in widget_html
    assert "Stop" in widget_html
    assert "2 events" in widget_html
    assert "16000 Hz" in widget_html


# --- Playback discipline: only one stream at a time ---


def test_stopPlayback_increments_generation(widget_html):
    """stopPlayback must increment playGen to cancel pending async decodes."""
    assert "playGen++;" in widget_html
    assert "function stopPlayback()" in widget_html


def test_stopPlayback_nulls_onended_before_stop(widget_html):
    """stopPlayback must null onended before calling stop() to prevent the
    stale callback from clobbering currentSource after a new source starts."""
    import re

    m = re.search(
        r"function stopPlayback\(\)\s*\{(.*?)\n  \}",
        widget_html,
        re.S,
    )
    assert m, "stopPlayback not found"
    body = m.group(1)
    # onended = null must appear before .stop()
    pos_null = body.index("onended = null")
    pos_stop = body.index(".stop()")
    assert (
        pos_null < pos_stop
    ), "onended must be nulled before stop() to prevent stale callback"


def test_playB64_checks_generation(widget_html):
    """playB64 async callback must bail out when generation is stale."""
    # playB64 should capture gen before decode and check it in callback
    assert "function playB64(" in widget_html
    assert "var gen = playGen;" in widget_html
    assert "if (gen !== playGen) return;" in widget_html


def test_playFullFrom_checks_generation(widget_html):
    """playFullFrom async callback must bail out when generation is stale."""
    assert "function playFullFrom(" in widget_html
    assert "if (gen !== playGen) return;" in widget_html


def test_all_play_paths_call_stopPlayback(widget_html):
    """Every play entry point must call stopPlayback first."""
    import re

    # playB64 body should start with stopPlayback()
    m = re.search(
        r"function playB64\([^)]*\)\s*\{(.*?)\n  \}", widget_html, re.S
    )
    assert m, "playB64 not found"
    assert "stopPlayback();" in m.group(1)

    # playFullFrom body should start with stopPlayback()
    m = re.search(
        r"function playFullFrom\([^)]*\)\s*\{(.*?)\n  \}", widget_html, re.S
    )
    assert m, "playFullFrom not found"
    assert "stopPlayback();" in m.group(1)


def test_stop_button_calls_stopPlayback(widget_html):
    """The Stop button handler must call stopPlayback."""
    assert 'btnStop.addEventListener("click"' in widget_html
    # The handler body includes stopPlayback()
    idx = widget_html.index('btnStop.addEventListener("click"')
    snippet = widget_html[idx : idx + 200]
    assert "stopPlayback();" in snippet


def test_ruler_click_calls_playFullFrom(widget_html):
    """Clicking the ruler must route through playFullFrom (single-play path)."""
    assert "isInRuler(pos)" in widget_html
    assert "playFullFrom(t);" in widget_html


def test_play_all_button_calls_playFullFrom(widget_html):
    """The Play all button routes through playFullFrom and supports pause."""
    idx = widget_html.index('btnPlay.addEventListener("click"')
    snippet = widget_html[idx : idx + 500]
    assert "playFullFrom(0);" in snippet  # play from start
    assert "playFullFrom(resumeFrom);" in snippet  # resume from pause
    assert "pausePlayback();" in snippet  # pause when playing


def test_detection_click_calls_playB64(widget_html):
    """Clicking a detection must route through playB64 (which calls stop)."""
    assert "playB64(regionB64[idx]" in widget_html


def test_pause_sets_button_text(widget_html):
    """pausePlayback and setPlaying toggle button innerHTML."""
    assert "function pausePlayback()" in widget_html
    assert "function setPlaying(" in widget_html
    # Pause label
    assert "Pause" in widget_html
    # Play label (restored on stop/end)
    assert "Play all" in widget_html


def test_stopPlayback_resets_pause_state(widget_html):
    """stopPlayback must clear pausedAt so next Play starts from beginning."""
    import re

    m = re.search(
        r"function stopPlayback\(\)\s*\{(.*?)\n  \}",
        widget_html,
        re.S,
    )
    assert m
    body = m.group(1)
    assert "pausedAt = -1" in body
    assert "setPlaying(false)" in body


def test_timestamp_display_element(widget_html):
    """Widget contains a selectable timestamp span updated during playback."""
    assert "timeSpan" in widget_html
    assert "user-select:all" in widget_html
    assert "updateTimestamp(" in widget_html
    assert "formatTimePrecise(" in widget_html


def test_energy_threshold_line():
    """When energy_threshold is provided, the widget draws a dashed line."""
    data = b"\x00\x10" * 800
    region = AudioRegion(data, 16000, 2, 1)
    events = [AudioRegion(data[:400], 16000, 2, 1, start=0.0)]
    html = _capture_widget_html(region, events)
    # No threshold by default
    assert "ethNorm" in html
    assert "[null]" in html

    # With threshold
    captured = []
    mock_display = MagicMock(side_effect=lambda obj: captured.append(obj))
    mock_html_cls = MagicMock(side_effect=lambda s: s)
    mock_mod = MagicMock()
    mock_mod.display = mock_display
    mock_mod.HTML = mock_html_cls
    with patch.dict("sys.modules", {"IPython.display": mock_mod}):
        display_interactive(region, events, energy_threshold=50)
    html_eth = captured[0]
    assert "ethNorm" in html_eth
    assert "[null]" not in html_eth  # should have a numeric value
    assert "setLineDash([6, 4])" in html_eth  # dashed line
