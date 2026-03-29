from pathlib import Path

import pytest

from auditok import AudioRegion, fix_pauses, make_silence, split
from auditok.io import get_audio_source
from auditok.util import AudioReader

MONO_RAW = "tests/data/test_split_10HZ_mono.raw"
STEREO_RAW = "tests/data/test_split_10HZ_stereo.raw"
SR, SW = 10, 2

SPLIT_KWARGS_10HZ = {
    "analysis_window": 0.1,
    "eth": 50,
}


def _load_raw(filename):
    with open(filename, "rb") as fp:
        return fp.read()


def _join_regions_with_silence(regions, silence_duration, sr, sw, ch):
    """Build expected output by manually joining regions with silence."""
    silence = make_silence(silence_duration, sr, sw, ch)
    return silence.join(regions)


# ── TypeError for fixed parameters ────────────────────────────────────


class TestFixPausesRejectsFixedParams:
    """max_dur and strict_min_dur must not be accepted."""

    def test_rejects_max_dur(self):
        with pytest.raises(TypeError, match="max_dur"):
            fix_pauses(MONO_RAW, 0.5, sr=SR, sw=SW, ch=1, max_dur=10)

    def test_rejects_strict_min_dur(self):
        with pytest.raises(TypeError, match="strict_min_dur"):
            fix_pauses(MONO_RAW, 0.5, sr=SR, sw=SW, ch=1, strict_min_dur=True)

    def test_rejects_both(self):
        with pytest.raises(TypeError, match="max_dur.*strict_min_dur"):
            fix_pauses(
                MONO_RAW,
                0.5,
                sr=SR,
                sw=SW,
                ch=1,
                max_dur=10,
                strict_min_dur=True,
            )


# ── Basic behavior with raw audio data ───────────────────────────────


class TestFixPausesBasic:
    """Core behavior: joins all detected events with fixed silence."""

    def test_basic_mono(self):
        """Result should match split(max_dur=None) joined with silence."""
        data = _load_raw(MONO_RAW)
        silence_dur = 0.5
        regions = list(
            split(
                data,
                sr=SR,
                sw=SW,
                ch=1,
                min_dur=0.2,
                max_dur=None,
                max_silence=0.3,
                **SPLIT_KWARGS_10HZ,
            )
        )
        expected = _join_regions_with_silence(regions, silence_dur, SR, SW, 1)

        result = fix_pauses(
            data,
            silence_dur,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert bytes(result) == bytes(expected)

    def test_basic_stereo(self):
        data = _load_raw(STEREO_RAW)
        silence_dur = 0.3
        regions = list(
            split(
                data,
                sr=SR,
                sw=SW,
                ch=2,
                min_dur=0.2,
                max_dur=None,
                max_silence=0.3,
                **SPLIT_KWARGS_10HZ,
            )
        )
        expected = _join_regions_with_silence(regions, silence_dur, SR, SW, 2)

        result = fix_pauses(
            data,
            silence_dur,
            sr=SR,
            sw=SW,
            ch=2,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert bytes(result) == bytes(expected)

    def test_zero_silence_duration(self):
        """With silence_duration=0, events should be concatenated."""
        data = _load_raw(MONO_RAW)
        regions = list(
            split(
                data,
                sr=SR,
                sw=SW,
                ch=1,
                min_dur=0.2,
                max_dur=None,
                max_silence=0.3,
                **SPLIT_KWARGS_10HZ,
            )
        )
        expected_data = b"".join(bytes(r) for r in regions)

        result = fix_pauses(
            data,
            0,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert bytes(result) == expected_data

    def test_preserves_audio_params(self):
        result = fix_pauses(
            MONO_RAW,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert result.sr == SR
        assert result.sw == SW
        assert result.ch == 1

    def test_preserves_audio_params_stereo(self):
        result = fix_pauses(
            STEREO_RAW,
            0.5,
            sr=SR,
            sw=SW,
            ch=2,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert result.sr == SR
        assert result.sw == SW
        assert result.ch == 2


# ── Events are never truncated (max_dur=None internally) ─────────────


class TestFixPausesNeverTruncates:
    """Unlike split(max_dur=5), fix_pauses never splits long events."""

    def test_long_event_not_split(self):
        """A long event that split(max_dur=5) would truncate stays intact."""
        data = _load_raw(MONO_RAW)
        # With max_dur=5 and max_silence=0.2, split gives 3 regions:
        # [(2,16), (17,31), (34,76)]
        regions_with_limit = list(
            split(
                data,
                sr=SR,
                sw=SW,
                ch=1,
                min_dur=0.2,
                max_dur=5,
                max_silence=0.2,
                **SPLIT_KWARGS_10HZ,
            )
        )
        # With max_dur=None and max_silence=0.2, split gives 3 regions too
        # but with different boundaries if max_dur was the limiting factor.
        # With max_silence=10, everything becomes one event.
        regions_no_limit = list(
            split(
                data,
                sr=SR,
                sw=SW,
                ch=1,
                min_dur=0.2,
                max_dur=None,
                max_silence=10,
                **SPLIT_KWARGS_10HZ,
            )
        )
        # max_silence=10 means everything is one event
        assert len(regions_no_limit) == 1
        assert len(regions_no_limit) < len(regions_with_limit)

        # fix_pauses with max_silence=10 should produce a single event
        result = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=10,
            **SPLIT_KWARGS_10HZ,
        )
        # Single event joined with nothing = just the event itself
        assert bytes(result) == bytes(regions_no_limit[0])

    def test_consistent_with_split_max_dur_none(self):
        """fix_pauses should give the same result as
        split_and_join_with_silence when max_dur=None."""
        from auditok import split_and_join_with_silence

        data = _load_raw(MONO_RAW)
        silence_dur = 0.5
        expected = split_and_join_with_silence(
            data,
            silence_dur,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_dur=None,
            max_silence=0.3,
            strict_min_dur=False,
            **SPLIT_KWARGS_10HZ,
        )
        result = fix_pauses(
            data,
            silence_dur,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert bytes(result) == bytes(expected)


# ── min_dur filters short events ──────────────────────────────────────


class TestFixPausesMinDur:
    """Short events below min_dur are discarded."""

    def test_short_events_removed(self):
        """With a high min_dur, short events should be dropped."""
        data = _load_raw(MONO_RAW)
        # With min_dur=0.2, max_silence=0.2, max_dur=None we get events.
        # With min_dur=3.0, only the long event (34,76) survives.
        regions_long_min = list(
            split(
                data,
                sr=SR,
                sw=SW,
                ch=1,
                min_dur=3.0,
                max_dur=None,
                max_silence=0.2,
                **SPLIT_KWARGS_10HZ,
            )
        )
        result = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=3.0,
            max_silence=0.2,
            **SPLIT_KWARGS_10HZ,
        )
        expected = _join_regions_with_silence(regions_long_min, 0.5, SR, SW, 1)
        assert bytes(result) == bytes(expected)
        # With min_dur=0.2 we get more events; with min_dur=3.0 fewer.
        regions_short_min = list(
            split(
                data,
                sr=SR,
                sw=SW,
                ch=1,
                min_dur=0.2,
                max_dur=None,
                max_silence=0.2,
                **SPLIT_KWARGS_10HZ,
            )
        )
        assert len(regions_long_min) < len(regions_short_min)

    def test_all_events_too_short_returns_empty(self):
        """If every event is below min_dur, return an empty AudioRegion."""
        data = _load_raw(MONO_RAW)
        result = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=100.0,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert isinstance(result, AudioRegion)
        assert not result
        assert result.duration == 0.0
        assert result.sr == SR
        assert result.sw == SW
        assert result.ch == 1


# ── No activity returns empty AudioRegion ─────────────────────────────


class TestFixPausesNoActivity:

    def test_no_activity_returns_empty(self):
        """High energy threshold means no detections."""
        data = _load_raw(MONO_RAW)
        result = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            analysis_window=0.1,
            eth=60,
        )
        assert isinstance(result, AudioRegion)
        assert not result
        assert result.duration == 0.0

    def test_pure_silence_returns_empty(self):
        """Input that is all zeros should yield no events."""
        silence_data = b"\0" * SR * SW * 1 * 10  # 10 seconds of silence
        result = fix_pauses(
            silence_data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert isinstance(result, AudioRegion)
        assert not result

    def test_empty_result_is_falsy(self):
        """Empty AudioRegion should evaluate to False like an empty list."""
        data = _load_raw(MONO_RAW)
        result = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=100.0,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert not result
        # Non-empty result should be truthy
        result_nonempty = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert result_nonempty


# ── max_leading_silence and max_trailing_silence ──────────────────────


class TestFixPausesSilenceParams:

    def test_max_trailing_silence_zero(self):
        """Trailing silence should be stripped from each event."""
        data = _load_raw(MONO_RAW)
        regions = list(
            split(
                data,
                sr=SR,
                sw=SW,
                ch=1,
                min_dur=0.2,
                max_dur=None,
                max_silence=0.3,
                max_trailing_silence=0,
                **SPLIT_KWARGS_10HZ,
            )
        )
        expected = _join_regions_with_silence(regions, 0.5, SR, SW, 1)

        result = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            max_trailing_silence=0,
            **SPLIT_KWARGS_10HZ,
        )
        assert bytes(result) == bytes(expected)

    def test_max_trailing_silence_changes_output(self):
        """Trimming trailing silence should produce shorter output."""
        data = _load_raw(MONO_RAW)
        result_with = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        result_trimmed = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            max_trailing_silence=0,
            **SPLIT_KWARGS_10HZ,
        )
        # Trimming trailing silence from events makes the total shorter
        assert result_trimmed.duration <= result_with.duration

    def test_max_leading_silence(self):
        """Leading silence should be prepended to each event."""
        data = _load_raw(MONO_RAW)
        result_no_lead = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            max_leading_silence=0,
            **SPLIT_KWARGS_10HZ,
        )
        result_with_lead = fix_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            max_leading_silence=0.2,
            **SPLIT_KWARGS_10HZ,
        )
        # Adding leading silence to events makes the total longer
        assert result_with_lead.duration >= result_no_lead.duration


# ── Silence duration variations ───────────────────────────────────────


class TestFixPausesSilenceDuration:

    @pytest.mark.parametrize(
        "silence_dur",
        [0, 0.1, 0.5, 1.0, 2.0],
        ids=["zero", "short", "medium", "one_sec", "long"],
    )
    def test_various_silence_durations(self, silence_dur):
        """Output should match manual join for any silence duration."""
        data = _load_raw(MONO_RAW)
        regions = list(
            split(
                data,
                sr=SR,
                sw=SW,
                ch=1,
                min_dur=0.2,
                max_dur=None,
                max_silence=0.3,
                **SPLIT_KWARGS_10HZ,
            )
        )
        expected = _join_regions_with_silence(regions, silence_dur, SR, SW, 1)

        result = fix_pauses(
            data,
            silence_dur,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        assert bytes(result) == bytes(expected)


# ── Input types ───────────────────────────────────────────────────────


class TestFixPausesInputTypes:
    """fix_pauses should work with all supported input types."""

    _COMMON_KW = {
        "min_dur": 0.2,
        "max_silence": 0.3,
        **SPLIT_KWARGS_10HZ,
    }

    def _reference(self):
        data = _load_raw(MONO_RAW)
        return bytes(
            fix_pauses(data, 0.5, sr=SR, sw=SW, ch=1, **self._COMMON_KW)
        )

    def test_filename(self):
        result = fix_pauses(
            MONO_RAW, 0.5, sr=SR, sw=SW, ch=1, **self._COMMON_KW
        )
        assert bytes(result) == self._reference()

    def test_path(self):
        result = fix_pauses(
            Path(MONO_RAW), 0.5, sr=SR, sw=SW, ch=1, **self._COMMON_KW
        )
        assert bytes(result) == self._reference()

    def test_bytes(self):
        data = _load_raw(MONO_RAW)
        result = fix_pauses(data, 0.5, sr=SR, sw=SW, ch=1, **self._COMMON_KW)
        assert bytes(result) == self._reference()

    def test_audio_region(self):
        data = _load_raw(MONO_RAW)
        region = AudioRegion(data, SR, SW, 1)
        result = fix_pauses(region, 0.5, **self._COMMON_KW)
        assert bytes(result) == self._reference()

    def test_audio_source(self):
        source = get_audio_source(MONO_RAW, sr=SR, sw=SW, ch=1)
        result = fix_pauses(source, 0.5, **self._COMMON_KW)
        assert bytes(result) == self._reference()

    def test_audio_reader(self):
        reader = AudioReader(
            MONO_RAW, sr=SR, sw=SW, ch=1, block_dur=0.1, record=True
        )
        result = fix_pauses(reader, 0.5, **self._COMMON_KW)
        assert bytes(result) == self._reference()
