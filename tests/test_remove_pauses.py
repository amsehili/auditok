from pathlib import Path

import pytest

from auditok import AudioRegion, make_silence, remove_pauses, split
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


class TestRemovePausesRejectsFixedParams:
    """max_dur and strict_min_dur must not be accepted."""

    def test_rejects_max_dur(self):
        with pytest.raises(TypeError, match="max_dur"):
            remove_pauses(MONO_RAW, 0.5, sr=SR, sw=SW, ch=1, max_dur=10)

    def test_rejects_strict_min_dur(self):
        with pytest.raises(TypeError, match="strict_min_dur"):
            remove_pauses(
                MONO_RAW, 0.5, sr=SR, sw=SW, ch=1, strict_min_dur=True
            )

    def test_rejects_both(self):
        with pytest.raises(TypeError, match="max_dur.*strict_min_dur"):
            remove_pauses(
                MONO_RAW,
                0.5,
                sr=SR,
                sw=SW,
                ch=1,
                max_dur=10,
                strict_min_dur=True,
            )


# ── Basic behavior with raw audio data ───────────────────────────────


class TestRemovePausesBasic:
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

        result = remove_pauses(
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

        result = remove_pauses(
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

        result = remove_pauses(
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
        result = remove_pauses(
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
        result = remove_pauses(
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


class TestRemovePausesNeverTruncates:
    """Unlike split(max_dur=5), remove_pauses never splits long events."""

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

        # remove_pauses with max_silence=10 should produce a single event
        result = remove_pauses(
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
        """remove_pauses should give the same result as
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
        result = remove_pauses(
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


class TestRemovePausesMinDur:
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
        result = remove_pauses(
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
        result = remove_pauses(
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


class TestRemovePausesNoActivity:

    def test_no_activity_returns_empty(self):
        """High energy threshold means no detections."""
        data = _load_raw(MONO_RAW)
        result = remove_pauses(
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
        result = remove_pauses(
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
        result = remove_pauses(
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
        result_nonempty = remove_pauses(
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


class TestRemovePausesSilenceParams:

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

        result = remove_pauses(
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
        result_with = remove_pauses(
            data,
            0.5,
            sr=SR,
            sw=SW,
            ch=1,
            min_dur=0.2,
            max_silence=0.3,
            **SPLIT_KWARGS_10HZ,
        )
        result_trimmed = remove_pauses(
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
        result_no_lead = remove_pauses(
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
        result_with_lead = remove_pauses(
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

    def test_drop_trailing_silence_deprecated(self):
        """drop_trailing_silence should trigger a DeprecationWarning."""
        data = _load_raw(MONO_RAW)
        with pytest.warns(DeprecationWarning, match="drop_trailing_silence"):
            remove_pauses(
                data,
                0.5,
                sr=SR,
                sw=SW,
                ch=1,
                min_dur=0.2,
                max_silence=0.3,
                drop_trailing_silence=True,
                **SPLIT_KWARGS_10HZ,
            )

    def test_drop_trailing_silence_equivalent_to_max_trailing_zero(self):
        """drop_trailing_silence=True should behave like max_trailing_silence=0."""
        data = _load_raw(MONO_RAW)
        with pytest.warns(DeprecationWarning):
            result_deprecated = remove_pauses(
                data,
                0.5,
                sr=SR,
                sw=SW,
                ch=1,
                min_dur=0.2,
                max_silence=0.3,
                drop_trailing_silence=True,
                **SPLIT_KWARGS_10HZ,
            )
        result_new = remove_pauses(
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
        assert bytes(result_deprecated) == bytes(result_new)


# ── Silence duration variations ───────────────────────────────────────


class TestRemovePausesSilenceDuration:

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

        result = remove_pauses(
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


class TestRemovePausesInputTypes:
    """remove_pauses should work with all supported input types."""

    _COMMON_KW = {
        "min_dur": 0.2,
        "max_silence": 0.3,
        **SPLIT_KWARGS_10HZ,
    }

    def _reference(self):
        data = _load_raw(MONO_RAW)
        return bytes(
            remove_pauses(data, 0.5, sr=SR, sw=SW, ch=1, **self._COMMON_KW)
        )

    def test_filename(self):
        result = remove_pauses(
            MONO_RAW, 0.5, sr=SR, sw=SW, ch=1, **self._COMMON_KW
        )
        assert bytes(result) == self._reference()

    def test_path(self):
        result = remove_pauses(
            Path(MONO_RAW), 0.5, sr=SR, sw=SW, ch=1, **self._COMMON_KW
        )
        assert bytes(result) == self._reference()

    def test_bytes(self):
        data = _load_raw(MONO_RAW)
        result = remove_pauses(data, 0.5, sr=SR, sw=SW, ch=1, **self._COMMON_KW)
        assert bytes(result) == self._reference()

    def test_audio_region(self):
        data = _load_raw(MONO_RAW)
        region = AudioRegion(data, SR, SW, 1)
        result = remove_pauses(region, 0.5, **self._COMMON_KW)
        assert bytes(result) == self._reference()

    def test_audio_source(self):
        source = get_audio_source(MONO_RAW, sr=SR, sw=SW, ch=1)
        result = remove_pauses(source, 0.5, **self._COMMON_KW)
        assert bytes(result) == self._reference()

    def test_audio_reader(self):
        reader = AudioReader(
            MONO_RAW, sr=SR, sw=SW, ch=1, block_dur=0.1, record=True
        )
        result = remove_pauses(reader, 0.5, **self._COMMON_KW)
        assert bytes(result) == self._reference()
