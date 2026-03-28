from pathlib import Path

import pytest

from auditok import AudioRegion, split, trim
from auditok.io import get_audio_source
from auditok.util import AudioReader

SPLIT_KWARGS_10HZ = {
    "min_dur": 0.2,
    "max_dur": 5,
    "max_silence": 0.2,
    "strict_min_dur": False,
    "analysis_window": 0.1,
    "eth": 50,
}

MONO_RAW = "tests/data/test_split_10HZ_mono.raw"
STEREO_RAW = "tests/data/test_split_10HZ_stereo.raw"
SR, SW = 10, 2


def _load_raw(filename):
    with open(filename, "rb") as fp:
        return fp.read()


def _expected_trimmed(data, channels, first_onset, last_offset):
    """Return the expected bytes for a trim between first_onset and
    last_offset (in samples)."""
    sample_size = SW * channels
    return data[first_onset * sample_size : last_offset * sample_size]


# ── AudioRegion.trim() ──────────────────────────────────────────────


class TestAudioRegionTrim:
    """Tests for AudioRegion.trim() method."""

    def test_trim_basic_mono(self):
        """Trim should remove leading and trailing silence."""
        data = _load_raw(MONO_RAW)
        region = AudioRegion(data, SR, SW, 1)
        # split with default params gives [(2,16), (17,31), (34,76)]
        # so trim should give data[2:76] (in samples)
        trimmed = region.trim(**SPLIT_KWARGS_10HZ)
        expected = _expected_trimmed(data, 1, 2, 76)
        assert bytes(trimmed) == expected

    def test_trim_basic_stereo(self):
        """Trim should work with stereo data."""
        data = _load_raw(STEREO_RAW)
        region = AudioRegion(data, SR, SW, 2)
        # stereo split with default params gives [(2,32), (34,76)]
        trimmed = region.trim(**SPLIT_KWARGS_10HZ)
        expected = _expected_trimmed(data, 2, 2, 76)
        assert bytes(trimmed) == expected

    def test_trim_preserves_audio_params(self):
        """Trimmed region should keep sr, sw, ch from the original."""
        data = _load_raw(STEREO_RAW)
        region = AudioRegion(data, SR, SW, 2)
        trimmed = region.trim(**SPLIT_KWARGS_10HZ)
        assert trimmed.sr == SR
        assert trimmed.sw == SW
        assert trimmed.ch == 2

    def test_trim_no_activity_returns_empty_region(self):
        """When no activity is detected, return an empty AudioRegion."""
        data = _load_raw(MONO_RAW)
        region = AudioRegion(data, SR, SW, 1)
        # energy_threshold=60 produces no detections on this data
        trimmed = region.trim(
            **{**SPLIT_KWARGS_10HZ, "eth": 60},
        )
        assert isinstance(trimmed, AudioRegion)
        assert trimmed.duration == 0.0
        assert trimmed.sr == SR
        assert trimmed.sw == SW
        assert trimmed.ch == 1

    def test_trim_empty_region_is_joinable(self):
        """Empty region from trim should be usable with + and join."""
        data = _load_raw(MONO_RAW)
        region = AudioRegion(data, SR, SW, 1)
        empty = region.trim(**{**SPLIT_KWARGS_10HZ, "eth": 60})
        # Should not raise
        joined = empty + region
        assert joined.duration == region.duration

    def test_trim_consistent_with_split(self):
        """trim() result should span from first split start to last split end."""
        data = _load_raw(MONO_RAW)
        region = AudioRegion(data, SR, SW, 1)
        regions = list(region.split(**SPLIT_KWARGS_10HZ))
        trimmed = region.trim(**SPLIT_KWARGS_10HZ)
        first, last = regions[0], regions[-1]
        expected = region.sec[first.start : last.end]
        assert bytes(trimmed) == bytes(expected)

    def test_trim_preserves_internal_silence(self):
        """Silence between detections must be preserved."""
        data = _load_raw(MONO_RAW)
        region = AudioRegion(data, SR, SW, 1)
        trimmed = region.trim(**SPLIT_KWARGS_10HZ)
        regions = list(region.split(**SPLIT_KWARGS_10HZ))
        # Sum of detection durations is less than trimmed duration
        # because internal silence is kept
        detection_dur = sum(r.duration for r in regions)
        assert trimmed.duration > detection_dur

    @pytest.mark.parametrize(
        "min_dur, max_dur, max_silence, max_trailing_silence, "
        "strict_min_dur, kwargs, expected_onset, expected_offset",
        [
            (0.2, 5, 0.2, None, False, {"eth": 50}, 2, 76),
            (3, 5, 0.2, None, False, {"eth": 50}, 34, 76),
            (0.2, 80, 10, None, False, {"eth": 50}, 2, 76),
            (0.2, 5, 0.2, 0, False, {"eth": 50}, 2, 76),
            (0.2, 5, 0.2, None, False, {"energy_threshold": 40}, 0, 76),
        ],
        ids=[
            "default",
            "long_min_dur",
            "long_max_silence",
            "drop_trailing_silence",
            "low_energy_threshold",
        ],
    )
    def test_trim_with_various_split_params(
        self,
        min_dur,
        max_dur,
        max_silence,
        max_trailing_silence,
        strict_min_dur,
        kwargs,
        expected_onset,
        expected_offset,
    ):
        """Trim boundaries should match first/last split region boundaries."""
        data = _load_raw(MONO_RAW)
        region = AudioRegion(data, SR, SW, 1)
        trim_kwargs = dict(
            min_dur=min_dur,
            max_dur=max_dur,
            max_silence=max_silence,
            strict_min_dur=strict_min_dur,
            analysis_window=0.1,
            **kwargs,
        )
        if max_trailing_silence is not None:
            trim_kwargs["max_trailing_silence"] = max_trailing_silence
        trimmed = region.trim(**trim_kwargs)
        expected = _expected_trimmed(data, 1, expected_onset, expected_offset)
        assert bytes(trimmed) == expected


# ── Module-level trim() ─────────────────────────────────────────────


class TestModuleTrim:
    """Tests for the module-level trim() function."""

    def test_trim_with_filename(self):
        data = _load_raw(MONO_RAW)
        trimmed = trim(MONO_RAW, sr=SR, sw=SW, ch=1, **SPLIT_KWARGS_10HZ)
        expected = _expected_trimmed(data, 1, 2, 76)
        assert bytes(trimmed) == expected

    def test_trim_with_path(self):
        data = _load_raw(MONO_RAW)
        trimmed = trim(Path(MONO_RAW), sr=SR, sw=SW, ch=1, **SPLIT_KWARGS_10HZ)
        expected = _expected_trimmed(data, 1, 2, 76)
        assert bytes(trimmed) == expected

    def test_trim_with_bytes(self):
        data = _load_raw(MONO_RAW)
        trimmed = trim(data, sr=SR, sw=SW, ch=1, **SPLIT_KWARGS_10HZ)
        expected = _expected_trimmed(data, 1, 2, 76)
        assert bytes(trimmed) == expected

    def test_trim_with_audio_region(self):
        data = _load_raw(MONO_RAW)
        region = AudioRegion(data, SR, SW, 1)
        trimmed = trim(region, **SPLIT_KWARGS_10HZ)
        expected = _expected_trimmed(data, 1, 2, 76)
        assert bytes(trimmed) == expected

    def test_trim_with_audio_source(self):
        data = _load_raw(MONO_RAW)
        source = get_audio_source(MONO_RAW, sr=SR, sw=SW, ch=1)
        trimmed = trim(source, **SPLIT_KWARGS_10HZ)
        expected = _expected_trimmed(data, 1, 2, 76)
        assert bytes(trimmed) == expected

    def test_trim_with_rewindable_audio_reader(self):
        """Rewindable AudioReader should be used directly (streaming)."""
        data = _load_raw(MONO_RAW)
        reader = AudioReader(
            MONO_RAW, sr=SR, sw=SW, ch=1, block_dur=0.1, record=True
        )
        trimmed = trim(reader, **SPLIT_KWARGS_10HZ)
        expected = _expected_trimmed(data, 1, 2, 76)
        assert bytes(trimmed) == expected

    def test_trim_with_non_rewindable_audio_reader(self):
        """Non-rewindable AudioReader should be re-wrapped with record=True."""
        data = _load_raw(MONO_RAW)
        reader = AudioReader(
            MONO_RAW, sr=SR, sw=SW, ch=1, block_dur=0.1, record=False
        )
        assert not reader.rewindable
        trimmed = trim(reader, **SPLIT_KWARGS_10HZ)
        expected = _expected_trimmed(data, 1, 2, 76)
        assert bytes(trimmed) == expected

    def test_trim_stereo(self):
        data = _load_raw(STEREO_RAW)
        trimmed = trim(STEREO_RAW, sr=SR, sw=SW, ch=2, **SPLIT_KWARGS_10HZ)
        expected = _expected_trimmed(data, 2, 2, 76)
        assert bytes(trimmed) == expected

    def test_trim_no_activity_returns_empty_region(self):
        data = _load_raw(MONO_RAW)
        trimmed = trim(
            data,
            sr=SR,
            sw=SW,
            ch=1,
            **{**SPLIT_KWARGS_10HZ, "eth": 60},
        )
        assert isinstance(trimmed, AudioRegion)
        assert trimmed.duration == 0.0

    def test_trim_consistent_with_audio_region_trim(self):
        """Module-level trim should give the same result as AudioRegion.trim."""
        data = _load_raw(MONO_RAW)
        region = AudioRegion(data, SR, SW, 1)
        trimmed_method = region.trim(**SPLIT_KWARGS_10HZ)
        trimmed_func = trim(data, sr=SR, sw=SW, ch=1, **SPLIT_KWARGS_10HZ)
        assert bytes(trimmed_func) == bytes(trimmed_method)

    def test_trim_consistent_across_input_types(self):
        """All input types should produce the same trimmed result."""
        data = _load_raw(STEREO_RAW)
        kw = {**SPLIT_KWARGS_10HZ}

        from_file = trim(STEREO_RAW, sr=SR, sw=SW, ch=2, **kw)
        from_path = trim(Path(STEREO_RAW), sr=SR, sw=SW, ch=2, **kw)
        from_bytes = trim(data, sr=SR, sw=SW, ch=2, **kw)
        from_region = trim(AudioRegion(data, SR, SW, 2), **kw)
        from_reader = trim(
            AudioReader(
                STEREO_RAW, sr=SR, sw=SW, ch=2, block_dur=0.1, record=True
            ),
            **kw,
        )
        from_source = trim(
            get_audio_source(STEREO_RAW, sr=SR, sw=SW, ch=2), **kw
        )

        reference = bytes(from_file)
        assert bytes(from_path) == reference
        assert bytes(from_bytes) == reference
        assert bytes(from_region) == reference
        assert bytes(from_reader) == reference
        assert bytes(from_source) == reference


# ── AudioReader wrapping AudioReader ─────────────────────────────────


class TestAudioReaderWrapping:
    """Tests for AudioReader accepting another AudioReader as input."""

    def test_audio_reader_wraps_non_rewindable(self):
        """AudioReader(AudioReader(record=False), record=True) should work."""
        data = _load_raw(MONO_RAW)
        inner = AudioReader(MONO_RAW, sr=SR, sw=SW, ch=1, block_dur=0.1)
        outer = AudioReader(inner, block_dur=0.1, record=True)
        outer.open()
        blocks = []
        while True:
            block = outer.read()
            if block is None:
                break
            blocks.append(block)
        outer.rewind()
        assert outer.data == data

    def test_audio_reader_wraps_rewindable(self):
        """AudioReader(AudioReader(record=True), record=True) should work."""
        data = _load_raw(MONO_RAW)
        inner = AudioReader(
            MONO_RAW, sr=SR, sw=SW, ch=1, block_dur=0.1, record=True
        )
        outer = AudioReader(inner, block_dur=0.1, record=True)
        outer.open()
        blocks = []
        while True:
            block = outer.read()
            if block is None:
                break
            blocks.append(block)
        outer.rewind()
        assert outer.data == data

    def test_wrapped_reader_preserves_audio_params(self):
        """Wrapped reader should expose the same sr, sw, ch."""
        inner = AudioReader(MONO_RAW, sr=SR, sw=SW, ch=1, block_dur=0.1)
        outer = AudioReader(inner, block_dur=0.1, record=True)
        assert outer.sr == SR
        assert outer.sw == SW
        assert outer.ch == 1
