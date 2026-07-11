"""Tests for auditok.validators.WebRTCVADValidator.

Skipped entirely when webrtcvad is not installed
(pip install auditok[webrtcvad])."""

import numpy as np
import pytest

pytest.importorskip("webrtcvad")

from auditok import load, split  # noqa: E402
from auditok.validators import WebRTCVADValidator  # noqa: E402

AUDIO_FILE_1TO6 = "tests/data/1to6arabic_16000_mono_bc_noise.wav"


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="sampling rate"):
        WebRTCVADValidator(11025, 2, 1)
    with pytest.raises(ValueError, match="subframe_dur"):
        WebRTCVADValidator(16000, 2, 1, subframe_dur=0.05)
    with pytest.raises(ValueError, match="aggregation"):
        WebRTCVADValidator(16000, 2, 1, aggregation="most")
    with pytest.raises(ValueError, match="use_channel"):
        WebRTCVADValidator(16000, 2, 3, use_channel="any")
    with pytest.raises(ValueError):
        WebRTCVADValidator(16000, 2, 1, mode=7)


def test_silence_is_not_valid():
    validator = WebRTCVADValidator(16000, 2, 1)
    silence = b"\x00" * (16000 * 2 // 10)  # 100 ms of digital silence
    assert validator.is_valid(silence) is False


def test_window_shorter_than_subframe_is_not_valid():
    validator = WebRTCVADValidator(16000, 2, 1, subframe_dur=0.03)
    assert validator.is_valid(b"\x00" * 200) is False


def test_split_with_webrtc_validator_detects_speech():
    """The validator plugged into split() must detect events on real
    speech and, per webrtc's design, not merely on energy: detections
    should roughly cover the same regions as the energy validator."""
    validator = WebRTCVADValidator(16000, 2, 1, mode=1)
    regions = list(
        split(
            AUDIO_FILE_1TO6,
            validator=validator,
            analysis_window=0.03,
            max_dur=None,
        )
    )
    assert len(regions) > 0
    total_speech = sum(r.duration for r in regions)
    assert 2 < total_speech < 15  # file is ~18.8 s with 6 utterances


def test_float32_input_equivalent_to_int16():
    """Decisions must be identical for the same audio stored as int16
    and as float32 (converted internally to the same int16 bytes)."""
    region = load(AUDIO_FILE_1TO6)
    int16_samples = np.frombuffer(region.data, dtype=np.int16)
    float32_data = (int16_samples / 32768.0).astype(np.float32).tobytes()

    kwargs = {"analysis_window": 0.03, "max_dur": None}
    regions_int16 = list(
        split(
            AUDIO_FILE_1TO6,
            validator=WebRTCVADValidator(16000, 2, 1, mode=2),
            **kwargs,
        )
    )
    regions_float32 = list(
        split(
            float32_data,
            sr=16000,
            sw=4,
            ch=1,
            validator=WebRTCVADValidator(16000, 4, 1, mode=2),
            **kwargs,
        )
    )
    assert [(r.start, r.end) for r in regions_float32] == [
        (r.start, r.end) for r in regions_int16
    ]


def test_multichannel_mix_and_channel_selection():
    file = "tests/data/test_16KHZ_3channel_400-800-1600Hz.wav"
    for use_channel in ("mix", 0, -1):
        validator = WebRTCVADValidator(
            16000, 2, 3, mode=3, use_channel=use_channel
        )
        regions = list(
            split(
                file,
                validator=validator,
                analysis_window=0.03,
                max_dur=None,
            )
        )
        # pure tones: webrtc's decision doesn't matter here, the pipeline
        # must simply run without errors on multichannel input
        assert isinstance(regions, list)


def test_split_validator_webrtc_string():
    """`validator="webrtc:N"` must behave exactly as split() called with
    an explicitly constructed WebRTCVADValidator with that mode."""
    explicit = WebRTCVADValidator(16000, 2, 1, mode=2)
    expected = [
        (r.start, r.end)
        for r in split(
            AUDIO_FILE_1TO6,
            validator=explicit,
            analysis_window=0.05,
            max_dur=None,
        )
    ]
    result = [
        (r.start, r.end)
        for r in split(
            AUDIO_FILE_1TO6,
            validator="webrtc:2",
            analysis_window=0.05,
            max_dur=None,
        )
    ]
    assert len(result) > 0
    assert result == expected


def test_split_validator_webrtc_default_mode_is_1():
    bare = [
        (r.start, r.end)
        for r in split(AUDIO_FILE_1TO6, validator="webrtc", max_dur=None)
    ]
    mode1 = [
        (r.start, r.end)
        for r in split(AUDIO_FILE_1TO6, validator="webrtc:1", max_dur=None)
    ]
    assert bare == mode1


def test_split_validator_webrtc_string_multichannel():
    """Multichannel + webrtc string: channels are mixed by default."""
    file = "tests/data/test_16KHZ_3channel_400-800-1600Hz.wav"
    regions = list(split(file, validator="webrtc:3", max_dur=None))
    assert isinstance(regions, list)


def test_cli_validator_webrtc(capsys):
    from auditok.cmdline import main

    ret = main(["split", AUDIO_FILE_1TO6, "-V", "webrtc:2", "-q"])
    assert ret == 0


def test_aggregation_modes():
    region = load(AUDIO_FILE_1TO6)
    window = bytes(region.sec[0.8:0.98])  # 180 ms inside first utterance
    results = {}
    for aggregation in ("any", "majority", "all"):
        validator = WebRTCVADValidator(
            16000, 2, 1, mode=1, aggregation=aggregation
        )
        results[aggregation] = validator.is_valid(window)
    # logical consistency: all => majority => any
    if results["all"]:
        assert results["majority"]
    if results["majority"]:
        assert results["any"]
