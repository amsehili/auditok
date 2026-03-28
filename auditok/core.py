"""
Module for main data structures and tokenization algorithms.

.. autosummary::
    :toctree: generated/

    load
    split
    make_silence
    split_and_join_with_silence
    AudioRegion
    StreamTokenizer
"""

from __future__ import annotations

import math
import os
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, Iterable

import numpy as np

from .exceptions import AudioParameterError, TooSmallBlockDuration
from .io import (
    AudioSource,
    check_audio_data,
    get_audio_source,
    player_for,
    to_file,
)
from .plotting import plot
from .util import AudioEnergyValidator, AudioReader, DataSource, DataValidator

try:
    from . import signal_numpy as signal  # type: ignore[attr-defined]
except ImportError:
    from . import signal

__all__ = [
    "load",
    "split",
    "trim",
    "make_silence",
    "split_and_join_with_silence",
    "AudioRegion",
    "StreamTokenizer",
]


DEFAULT_ANALYSIS_WINDOW = 0.05
DEFAULT_ENERGY_THRESHOLD = 50
_EPSILON = 1e-10

_DROP_TRAILING_SILENCE_DEPRECATION = (
    "'drop_trailing_silence' is deprecated and will be removed in a "
    "future version. Use max_trailing_silence=0 to drop trailing "
    "silence, or max_trailing_silence=None (default) to keep all "
    "trailing silence (determined by max_silence)."
)


def load(
    input: str | Path | bytes | AudioSource | None,
    skip: float = 0,
    max_read: float | None = None,
    **kwargs: Any,
) -> AudioRegion:
    """
    Load audio data from a specified source and return it as an
    :class:`AudioRegion`.

    Parameters
    ----------
    input : None, str, Path, bytes, AudioSource
        The source from which to read audio data. If a `str` or `Path`, it
        should specify the path to a valid audio file. If `bytes`, it is
        treated as raw audio data. If set to "-", raw data will be read from
        standard input (stdin). If `None`, audio data is read from the
        microphone using sounddevice. For `bytes` data or a raw audio file path,
        `sampling_rate`, `sample_width`, and `channels` parameters (or their
        aliases) must be specified. If an :class:`AudioSource` object is
        provided, it is used directly to read data.
    skip : float, default: 0
        Duration in seconds of audio data to skip from the beginning of the
        source. When reading from a microphone, `skip` must be 0; otherwise,
        a `ValueError` is raised.
    max_read : float, default: None
        Duration in seconds of audio data to read from the source. When reading
        from the microphone, `max_read` must not be `None`; otherwise, a
        `ValueError` is raised.
    audio_format, fmt : str
        Format of the audio data (e.g., wav, ogg, flac, raw, etc.). This is
        only used if `input` is a string path to an audio file. If not
        provided, the audio format is inferred from the file's extension or
        header.
    sampling_rate, sr : int
        Sampling rate of the audio data. Required if `input` is a raw audio
        file, a `bytes` object, or `None` (i.e., when reading from the
        microphone).
    sample_width, sw : int
        Number of bytes used to encode a single audio sample, typically 1, 2,
        or 4. Required for raw audio data; see `sampling_rate`.
    channels, ch : int
        Number of channels in the audio data. Required for raw audio data;
        see `sampling_rate`.
    large_file : bool, default: False
        If `True`, and `input` is a path to a *wav* or *raw* audio file, the
        file is not fully loaded into memory to create the region (only the
        necessary portion of data is loaded). This should be set to `True`
        when `max_read` is much smaller than the total size of a large audio
        file, to avoid loading the entire file into memory.

    Returns
    -------
    region : AudioRegion

    Raises
    ------
    ValueError
        Raised if `input` is `None` (i.e., reading from the microphone) and
        `skip` is not 0, or if `max_read` is `None` when `input` is `None`.
        This ensures that when reading from the microphone, no data is
        skipped, and the maximum amount of data to read is explicitly
        specified.
    """

    return AudioRegion.load(input, skip, max_read, **kwargs)


def split(
    input: str | Path | bytes | AudioSource | AudioReader | AudioRegion | None,
    *,
    min_dur: float = 0.2,
    max_dur: float | None = 5,
    max_silence: float = 0.3,
    max_leading_silence: float = 0,
    max_trailing_silence: float | None = None,
    drop_trailing_silence: bool | None = None,
    strict_min_dur: bool = False,
    **kwargs: Any,
) -> Generator[AudioRegion, None, None]:
    """
    Split audio data and return a generator of :class:`AudioRegion`s.

    Parameters
    ----------
    input : str, Path, bytes, AudioSource, AudioReader, AudioRegion, or None
        Audio data input. If `str` or `Path`, it should be the path to an audio
        file. Use "-" to indicate standard input. If bytes, the input is treated
        as raw audio data. If None, audio is read from the microphone.

        Any input not of type `AudioReader` is converted into an `AudioReader`
        before processing. If `input` is raw audio data (str, bytes, or None),
        specify audio parameters using kwargs (e.g., `sampling_rate`,
        `sample_width`, `channels`).

        For string inputs, audio format is inferred from the file extension, or
        specify explicitly via `audio_format` or `fmt`. For compressed formats,
        ffmpeg is used to decode the audio.

    min_dur : float, default=0.2
        Minimum duration in seconds of a detected audio event. Higher values
        can exclude very short utterances (e.g., single words like "yes" or
        "no"). Lower values may increase the number of short audio events.

    max_dur : float or None, default=5
        Maximum duration in seconds for an audio event. Events longer than this
        are truncated. If the remainder of a truncated event is shorter than
        `min_dur`, it is included as a valid event if `strict_min_dur` is False;
        otherwise, it is rejected. Use None to disable the maximum duration
        limit (i.e., events can be of arbitrary length).

    max_silence : float, default=0.3
        Maximum duration of continuous silence allowed within an audio event.
        Multiple silent gaps of this duration may appear in a single event.
        This controls *when* a detection ends; use ``max_trailing_silence``
        to control how much of that trailing silence to keep.

    max_leading_silence : float, default=0
        Maximum duration in seconds of non-valid audio (silence) to retain
        immediately before each detected event. When an audio event is
        detected, up to `max_leading_silence` seconds of the preceding
        silence are prepended to the event, and its start time is adjusted
        backward accordingly.

        This is useful for audio events where the onset energy rises
        gradually — for example, speech utterances whose initial frames
        fall below the energy threshold. Without leading silence, those
        frames are cut, producing an abrupt start that can sound
        uncomfortable. Including a small amount of leading context (e.g.,
        0.1-0.3 seconds) preserves the natural attack of the sound.

        When set to 0 (default), events start at the first valid frame.

    max_trailing_silence : float or None, default=None
        Maximum duration in seconds of trailing silence to retain at the
        end of each detected event. When an event ends, up to
        ``max_trailing_silence`` seconds of the accumulated trailing
        silence are kept; any excess is trimmed.

        - ``None`` (default): keep all trailing silence (up to
          ``max_silence``). This preserves the natural decay of the sound.
        - ``0``: drop all trailing silence.
        - A positive value: keep up to that many seconds of trailing
          silence, independently of ``max_silence``.

    drop_trailing_silence : bool or None, default=None
        .. deprecated::
            Use ``max_trailing_silence=0`` instead of
            ``drop_trailing_silence=True``, or ``max_trailing_silence=None``
            (default) to keep all trailing silence (determined by
            ``max_silence``).

    strict_min_dur : bool, default=False
        Whether to strictly enforce `min_dur` for all events, rejecting any
        event shorter than `min_dur`, even if contiguous with a valid event.

    Other Parameters
    ----------------
    analysis_window, aw : float, default=0.05 (50 ms)
        Duration of analysis window in seconds. Values between 0.01 and 0.1 are
        generally effective.

    audio_format, fmt : str
        Type of audio data (e.g., wav, ogg, flac, raw). Used if `input` is a
        file path. If not specified, audio format is inferred from the file
        extension or header.

    sampling_rate, sr : int
        Sampling rate of audio data, required if `input` is raw data (bytes or
        None).

    sample_width, sw : int
        Number of bytes per audio sample (typically 1, 2, or 4). Required for
        raw audio; see `sampling_rate`.

    channels, ch : int
        Number of audio channels. Required for raw data; see `sampling_rate`.

    use_channel, uc : {None, "mix"} or int
        Channel selection for splitting if `input` has multiple channels. All
        channels are retained in detected events. Options:

        - None or "any" (default): accept activity from any channel.
        - "mix" or "average": mix all channels into a single averaged channel.
        - int (0 <= value < channels): use the specified channel ID for splitting.

    large_file : bool, default=False
        If True and `input` is a path to a wav or raw file, audio is processed
        lazily. Otherwise, the entire file is loaded before splitting. Set to
        True if file size exceeds available memory.

    max_read, mr : float, default=None
        Maximum data read from source in seconds. Default is to read to end.

    validator, val : callable or DataValidator, default=None
        Custom validator for audio data. If None, uses `AudioEnergyValidator`
        with the given `energy_threshold`. Should be callable or an instance of
        `DataValidator` implementing `is_valid`.

    energy_threshold, eth : float, default=50
        Energy threshold for audio activity detection. Audio regions with
        sufficient signal energy above this threshold are considered valid.
        Calculated as the log energy: `20 * log10(sqrt(dot(x, x) / len(x)))`.
        Ignored if `validator` is specified.

    Yields
    ------
    AudioRegion
        Generator yielding detected :class:`AudioRegion` instances.
    """

    if min_dur <= 0:
        raise ValueError(f"'min_dur' ({min_dur}) must be > 0")
    if max_dur is None or max_dur == float("inf"):
        max_dur = float("inf")
    elif max_dur <= 0:
        raise ValueError(f"'max_dur' ({max_dur}) must be > 0")
    if max_silence < 0:
        raise ValueError(f"'max_silence' ({max_silence}) must be >= 0")

    if isinstance(input, AudioReader):
        source = input
    else:
        analysis_window = kwargs.get(
            "analysis_window", kwargs.get("aw", DEFAULT_ANALYSIS_WINDOW)
        )
        if analysis_window <= 0:
            raise ValueError(
                f"'analysis_window' ({analysis_window}) must be > 0"
            )

        params = kwargs.copy()
        params["max_read"] = params.get("max_read", params.get("mr"))
        params["audio_format"] = params.get("audio_format", params.get("fmt"))
        if isinstance(input, AudioRegion):
            params["sampling_rate"] = input.sr
            params["sample_width"] = input.sw
            params["channels"] = input.ch
            input = bytes(input)
        try:
            source = AudioReader(input, block_dur=analysis_window, **params)
        except TooSmallBlockDuration as exc:
            err_msg = f"Too small 'analysis_window' ({exc.block_dur}) for "
            err_msg += f"sampling rate ({exc.sampling_rate}). Analysis window "
            err_msg += f"should at least be 1/{exc.sampling_rate} to cover "
            err_msg += "one data sample"
            raise ValueError(err_msg) from exc
    analysis_window = source.block_dur

    validator = kwargs.get("validator", kwargs.get("val"))
    if validator is None:
        energy_threshold = kwargs.get(
            "energy_threshold", kwargs.get("eth", DEFAULT_ENERGY_THRESHOLD)
        )
        use_channel = kwargs.get("use_channel", kwargs.get("uc"))
        validator = AudioEnergyValidator(
            energy_threshold, source.sw, source.ch, use_channel=use_channel
        )
    # Handle deprecated drop_trailing_silence
    if drop_trailing_silence is not None:
        warnings.warn(
            _DROP_TRAILING_SILENCE_DEPRECATION,
            DeprecationWarning,
            stacklevel=2,
        )
        if max_trailing_silence is None:
            max_trailing_silence = 0.0 if drop_trailing_silence else None

    mode = StreamTokenizer.STRICT_MIN_LENGTH if strict_min_dur else 0
    min_length = _duration_to_nb_windows(min_dur, analysis_window, math.ceil)
    if max_dur == float("inf"):
        max_length = float("inf")
    else:
        max_length = _duration_to_nb_windows(
            max_dur, analysis_window, math.floor, _EPSILON
        )
    max_continuous_silence = _duration_to_nb_windows(
        max_silence, analysis_window, math.floor, _EPSILON
    )
    max_leading_silence_frames = _duration_to_nb_windows(
        max_leading_silence, analysis_window, math.floor, _EPSILON
    )
    if max_trailing_silence is not None:
        max_trailing_silence_frames: int | None = _duration_to_nb_windows(
            max_trailing_silence, analysis_window, math.floor, _EPSILON
        )
    else:
        max_trailing_silence_frames = None

    err_msg = "({0} sec.) results in {1} analysis window(s) "
    err_msg += "({1} == {6}({0} / {2})) which is {5} the number "
    err_msg += "of analysis window(s) for 'max_dur' "
    err_msg += "({3} == floor({4} / {2}))"
    if min_length > max_length:
        err_msg = "'min_dur' " + err_msg
        raise ValueError(
            err_msg.format(
                min_dur,
                min_length,
                analysis_window,
                max_length,
                max_dur,
                "higher than",
                "ceil",
            )
        )

    if max_continuous_silence >= max_length:
        err_msg = "'max_silence' " + err_msg
        raise ValueError(
            err_msg.format(
                max_silence,
                max_continuous_silence,
                analysis_window,
                max_length,
                max_dur,
                "higher or equal to",
                "floor",
            )
        )

    tokenizer = StreamTokenizer(
        validator,
        min_length,
        max_length,
        max_continuous_silence,
        mode=mode,
        max_leading_silence=max_leading_silence_frames,
        max_trailing_silence=max_trailing_silence_frames,
    )
    source.open()
    token_gen = tokenizer.tokenize(source, generator=True)
    region_gen = (
        _make_audio_region(
            token[0],
            token[1],
            source.block_dur,
            source.sr,
            source.sw,
            source.ch,
        )
        for token in token_gen
    )
    return region_gen


def make_silence(
    duration: float,
    sampling_rate: int = 16000,
    sample_width: int = 2,
    channels: int = 1,
) -> AudioRegion:
    """
    Generate a silence of specified duration.

    Parameters
    ----------
    duration : float
        Duration of silence in seconds.
    sampling_rate : int, optional
        Sampling rate of the audio data, default is 16000.
    sample_width : int, optional
        Number of bytes per audio sample, default is 2.
    channels : int, optional
        Number of audio channels, default is 1.

    Returns
    -------
    AudioRegion
        A "silent" AudioRegion of the specified duration.
    """
    size = round(duration * sampling_rate) * sample_width * channels
    data = b"\0" * size
    region = AudioRegion(data, sampling_rate, sample_width, channels)
    return region


def split_and_join_with_silence(
    input: str | Path | bytes | AudioSource | AudioReader | AudioRegion | None,
    silence_duration: float,
    **kwargs: Any,
) -> AudioRegion | None:
    """
    Split input audio and join (glue) the resulting regions with a specified
    silence duration between them. This can be used to adjust the length of
    silence between audio events, either shortening or lengthening pauses.

    Parameters
    ----------
    input : str, Path, bytes, AudioSource, AudioReader, AudioRegion, or None
        Audio input (see :func:`split` for details).
    silence_duration : float
        Duration of silence in seconds between audio events.

    **kwargs
        Additional parameters forwarded to :func:`split` and
        :class:`AudioReader` (e.g., ``energy_threshold``,
        ``analysis_window``, ``sampling_rate``, ``sample_width``,
        ``channels``, ``audio_format``).

    Returns
    -------
    AudioRegion or None
        An :meth:`AudioRegion` with the specified between-events silence
        duration. Returns None if no audio events are detected in the input
        data.
    """
    regions = list(split(input, **kwargs))
    if regions:
        first = regions[0]
        # create a silence with the same parameters as input audio
        silence = make_silence(silence_duration, first.sr, first.sw, first.ch)
        return silence.join(regions)
    return None


def trim(
    input: str | Path | bytes | AudioSource | AudioReader | AudioRegion | None,
    *,
    min_dur: float = 0.2,
    max_dur: float | None = 5,
    max_silence: float = 0.3,
    max_leading_silence: float = 0,
    max_trailing_silence: float | None = None,
    drop_trailing_silence: bool | None = None,
    strict_min_dur: bool = False,
    **kwargs: Any,
) -> AudioRegion:
    """
    Detect audio activity in `input` and return the audio between the start
    of the first detection and the end of the last, removing leading and
    trailing silence.

    For non-AudioRegion inputs (files, microphone, stdin), data is recorded
    as :func:`split` consumes it, so the stream is only read once. This
    makes ``trim`` efficient for live sources where a second pass is
    impossible or expensive.

    Parameters
    ----------
    input : str, Path, bytes, AudioSource, AudioReader, AudioRegion, or None
        Audio input (see :func:`split` for details).

    See :func:`split` for descriptions of all other parameters.

    Returns
    -------
    AudioRegion
        The trimmed audio region. Returns an empty AudioRegion (zero
        duration) if no audio activity is detected.

    See Also
    --------
    AudioRegion.trim : Trim an in-memory AudioRegion.
    split : Split audio into individual activity regions.
    """
    if drop_trailing_silence is not None:
        warnings.warn(
            _DROP_TRAILING_SILENCE_DEPRECATION,
            DeprecationWarning,
            stacklevel=2,
        )
        if max_trailing_silence is None:
            max_trailing_silence = 0.0 if drop_trailing_silence else None
    if isinstance(input, AudioRegion):
        return input.trim(
            min_dur=min_dur,
            max_dur=max_dur,
            max_silence=max_silence,
            max_leading_silence=max_leading_silence,
            max_trailing_silence=max_trailing_silence,
            strict_min_dur=strict_min_dur,
            **kwargs,
        )

    # Wrap input in a recording AudioReader so data is cached as split()
    # consumes it block by block. This avoids a second pass over the source
    # (critical for microphone / stdin where data can't be re-read).
    if not isinstance(input, AudioReader):
        analysis_window = kwargs.get(
            "analysis_window", kwargs.get("aw", DEFAULT_ANALYSIS_WINDOW)
        )
        reader = AudioReader(
            input, block_dur=analysis_window, record=True, **kwargs
        )
    elif input.rewindable:
        reader = input
    else:
        # Non-rewindable AudioReader: re-wrap with record=True so data
        # is cached during the single streaming pass through split().
        reader = AudioReader(input, block_dur=input.block_dur, record=True)

    first = None
    last = None
    for region in split(
        reader,
        min_dur=min_dur,
        max_dur=max_dur,
        max_silence=max_silence,
        max_leading_silence=max_leading_silence,
        max_trailing_silence=max_trailing_silence,
        strict_min_dur=strict_min_dur,
        **kwargs,
    ):
        if first is None:
            first = region
        last = region

    if first is None:
        reader.close()
        return AudioRegion(b"", reader.sr, reader.sw, reader.ch)

    reader.rewind()
    full_region = AudioRegion(reader.data, reader.sr, reader.sw, reader.ch)
    reader.close()
    return full_region.sec[first.start : last.end]  # type: ignore[misc]


def _duration_to_nb_windows(
    duration, analysis_window, round_fn=round, epsilon=0
):
    """
    Helper function to convert a given duration into a positive integer
    of analysis windows. If `duration / analysis_window` is not an integer,
    the result will be rounded up to the nearest integer. If `duration == 0`,
    returns 0. If `duration < analysis_window`, returns 1.

    Both `duration` and `analysis_window` should be in the same units,
    either seconds or milliseconds.

    Parameters
    ----------
    duration : float
        The given duration in seconds or milliseconds.
    analysis_window : float
        The size of each analysis window, in the same units as `duration`.
    round_fn : callable, optional
        A function for rounding the result, default is `round`.
    epsilon : float, optional
        A small value added before rounding to address floating-point
        precision issues, ensuring accurate rounding for cases like
        `0.3 / 0.1`, where `round_fn=math.floor` would otherwise yield
        an incorrect result.

    Returns
    -------
    nb_windows : int
        The minimum number of `analysis_window` units needed to cover
        `duration`, ensuring `analysis_window * nb_windows >= duration`.
    """

    if duration < 0 or analysis_window <= 0:
        err_msg = "'duration' ({}) must be >= 0 and 'analysis_window' ({}) > 0"
        raise ValueError(err_msg.format(duration, analysis_window))
    if duration == 0:
        return 0
    return int(round_fn(duration / analysis_window + epsilon))


def _make_audio_region(
    data_frames,
    start_frame,
    frame_duration,
    sampling_rate,
    sample_width,
    channels,
):
    """
    Helper function to create an :class:`AudioRegion` from parameters provided
    by a tokenization object. This function handles setting the `start` and `end`
    metadata for the region.

    Parameters
    ----------
    frame_duration : float
        Duration of each analysis window in seconds.
    start_frame : int
        Index of the first analysis window.
    sampling_rate : int
        Sampling rate of the audio data.
    sample_width : int
        Number of bytes per audio sample.
    channels : int
        Number of audio channels.

    Returns
    -------
    audio_region : AudioRegion
        An AudioRegion object with `start` time calculated as:
        `1000 * start_frame * frame_duration`.
    """
    start = start_frame * frame_duration
    data = b"".join(data_frames)
    return AudioRegion(data, sampling_rate, sample_width, channels, start)


def _read_chunks_online(max_read, **kwargs):
    """
    Helper function to read audio data from an online blocking source
    (e.g., a microphone). This function builds an `AudioRegion` and can
    intercept `KeyboardInterrupt` to stop reading immediately when the
    exception is raised, making it more user-friendly for [i]Python sessions
    and Jupyter notebooks.

    Parameters
    ----------
    max_read : float
        Maximum duration of audio data to read, in seconds.
    kwargs :
        Additional audio parameters (e.g., `sampling_rate`, `sample_width`,
        and `channels`).
    """
    reader = AudioReader(None, block_dur=0.5, max_read=max_read, **kwargs)
    reader.open()
    data = []
    try:
        while True:
            frame = reader.read()
            if frame is None:
                break
            data.append(frame)
    except KeyboardInterrupt:
        # Stop data acquisition from microphone when pressing
        # Ctrl+C in an [i]python session or a notebook
        pass
    reader.close()
    return (
        b"".join(data),
        reader.sampling_rate,
        reader.sample_width,
        reader.channels,
    )


def _read_offline(input, skip=0, max_read=None, **kwargs):
    """
    Helper function to read audio data from an offline source (e.g., file).
    This function is used to build :class:`AudioRegion` objects.

    Parameters
    ----------
    input : str or bytes
        Path to an audio file (if str) or a bytes object representing raw
        audio data.
    skip : float, optional, default=0
        Amount of data to skip from the beginning of the audio source, in
        seconds.
    max_read : float, optional, default=None
        Maximum duration of audio data to read, in seconds. Default is None,
        which reads until the end of the stream.
    kwargs :
        Additional audio parameters (e.g., `sampling_rate`, `sample_width`,
        and `channels`).
    """

    audio_source = get_audio_source(input, **kwargs)
    audio_source.open()
    if skip is not None and skip > 0:
        skip_samples = round(skip * audio_source.sampling_rate)
        audio_source.read(skip_samples)
    if max_read is not None:
        if max_read < 0:
            max_read = None
        else:
            max_read = round(max_read * audio_source.sampling_rate)
    data = audio_source.read(max_read)
    audio_source.close()
    return (
        data,
        audio_source.sampling_rate,
        audio_source.sample_width,
        audio_source.channels,
    )


def _check_convert_index(index, types, err_msg):
    if not isinstance(index, slice) or index.step is not None:
        raise TypeError(err_msg)
    start = index.start if index.start is not None else 0
    stop = index.stop
    for index in (start, stop):
        if index is not None and not isinstance(index, types):
            raise TypeError(err_msg)
    return start, stop


class _SecondsView:
    """
    A class to create a view of an :class:`AudioRegion` that supports slicing
    with time-based indices in seconds.
    """

    def __init__(self, region: AudioRegion) -> None:
        self._region = region

    def __getitem__(self, index: slice) -> AudioRegion:
        err_msg = "Slicing AudioRegion by seconds requires indices of type "
        err_msg += "'int' or 'float' without a step (e.g. region.sec[7.5:10])"
        start_s, stop_s = _check_convert_index(index, (int, float), err_msg)
        sr = self._region.sampling_rate
        start_sample = int(start_s * sr)
        stop_sample = None if stop_s is None else round(stop_s * sr)
        return self._region[start_sample:stop_sample]

    @property
    def len(self):
        """
        Return region duration in seconds.
        """
        return self._region.duration


class _MillisView(_SecondsView):
    """A class to create a view of `AudioRegion` that can be sliced using
    indices in milliseconds.
    """

    def __getitem__(self, index):
        err_msg = (
            "Slicing AudioRegion by milliseconds requires indices of type "
        )
        err_msg += "'int' without a step (e.g. region.sec[500:1500])"
        start_ms, stop_ms = _check_convert_index(index, (int), err_msg)
        start_sec = start_ms / 1000
        stop_sec = None if stop_ms is None else stop_ms / 1000
        index = slice(start_sec, stop_sec)
        return super().__getitem__(index)

    def __len__(self):
        """
        Return region duration in milliseconds.
        """
        return round(self._region.duration * 1000)

    @property
    def len(self):
        """
        Return region duration in milliseconds.
        """
        return len(self)


@dataclass(frozen=True)
class AudioRegion(object):
    """
    `AudioRegion` encapsulates raw audio data and provides an interface for
    performing basic audio operations. Use :meth:`AudioRegion.load` or
    :func:`load` to create an `AudioRegion` from various input types.

    Parameters
    ----------
    data : bytes
        Raw audio data as a bytes object.
    sampling_rate : int
        Sampling rate of the audio data.
    sample_width : int
        Number of bytes per audio sample.
    channels : int
        Number of audio channels.
    start : float, optional, default=None
        Optional start time of the region, typically provided by the `split`
        function.
    """

    data: bytes
    sampling_rate: int
    sample_width: int
    channels: int
    start: float | None = field(default=None, repr=None)

    def __post_init__(self) -> None:

        check_audio_data(self.data, self.sample_width, self.channels)
        object.__setattr__(self, "splitp", self.split_and_plot)
        duration = len(self.data) / (
            self.sampling_rate * self.sample_width * self.channels
        )
        object.__setattr__(self, "duration", duration)

        if self.start is not None:
            object.__setattr__(self, "end", self.start + self.duration)
        else:
            object.__setattr__(self, "end", None)

        # `seconds` and `millis` are defined below as @property with docstring
        object.__setattr__(self, "_seconds_view", _SecondsView(self))
        object.__setattr__(self, "_millis_view", _MillisView(self))

        object.__setattr__(self, "sec", self.seconds)
        object.__setattr__(self, "s", self.seconds)
        object.__setattr__(self, "ms", self.millis)

    @classmethod
    def load(
        cls,
        input: str | Path | bytes | AudioSource | None,
        skip: float = 0,
        max_read: float | None = None,
        **kwargs: Any,
    ) -> AudioRegion:
        """
        Create an :class:`AudioRegion` by loading data from `input`.

        See :func:`load` for a full description of parameters.

        Returns
        -------
        region : AudioRegion
            An AudioRegion instance created from the specified input data.

        Raises
        ------
        ValueError
            Raised if `input` is None and either `skip` is not 0 or `max_read`
            is None.
        """

        if input is None:
            if skip > 0:
                raise ValueError(
                    "'skip' should be 0 when reading from microphone"
                )
            if max_read is None or max_read < 0:
                raise ValueError(
                    "'max_read' should not be None when reading from "
                    "microphone"
                )
            data, sampling_rate, sample_width, channels = _read_chunks_online(
                max_read, **kwargs
            )
        else:
            data, sampling_rate, sample_width, channels = _read_offline(
                input, skip=skip, max_read=max_read, **kwargs
            )

        return cls(data, sampling_rate, sample_width, channels)

    @property
    def seconds(self) -> _SecondsView:
        """
        A view to slice audio region by seconds using
        ``region.seconds[start:end]``.
        """
        return self._seconds_view

    @property
    def millis(self) -> _MillisView:
        """A view to slice audio region by milliseconds using
        ``region.millis[start:end]``."""
        return self._millis_view

    @property
    def sr(self) -> int:
        """Sampling rate of audio data, alias for `sampling_rate`."""
        return self.sampling_rate

    @property
    def sw(self) -> int:
        """Number of bytes per sample, alias for `sample_width`."""
        return self.sample_width

    @property
    def ch(self) -> int:
        """Number of channels of audio data, alias for `channels`."""
        return self.channels

    def play(
        self,
        progress_bar: bool = False,
        player: Any = None,
        **progress_bar_kwargs: Any,
    ) -> None:
        """
        Play the audio region.

        Parameters
        ----------
        progress_bar : bool, optional, default=False
            Whether to display a progress bar during playback. Requires `tqdm`,
            if not installed, no progress bar will be shown.
        player : AudioPlayer, optional, default=None
            Audio player to use for playback. If None (default), a new player is
            obtained via `player_for()`.
        progress_bar_kwargs : dict, optional
            Additional keyword arguments to pass to the `tqdm` progress bar
            (e.g., `leave=False` to clear the bar from the screen upon completion).
        """

        if player is None:
            player = player_for(self)
        player.play(self.data, progress_bar=progress_bar, **progress_bar_kwargs)

    def save(
        self,
        filename: str | Path,
        audio_format: str | None = None,
        exists_ok: bool = True,
        **audio_parameters: Any,
    ) -> str | Path:
        """
        Save the audio region to a file.

        For raw and wav formats, the region's audio parameters are used as-is.
        For compressed formats (mp3, ogg, flac, etc.), ffmpeg is used as the
        encoding backend.

        When ``audio_codec``, ``audio_bitrate``, ``audio_quality``, or
        ``ffmpeg_extra_args`` are provided, they are forwarded to ffmpeg to
        control the encoding.

        Parameters
        ----------
        filename : str or Path
            Path to the output audio file. If a string, it may include
            ``{start}``, ``{end}``, and ``{duration}`` placeholders. Regions
            created by :meth:`split` contain ``start`` and ``end`` attributes
            that can be used to format the filename, as shown in the example.
        audio_format : str, optional, default=None
            Format used to save the audio data (e.g., "wav", "mp3", "ogg",
            "flac"). If None (default), the format is inferred from the file
            extension. If the filename has no extension, the audio is saved as
            a raw (headerless) audio file.
        exists_ok : bool, optional, default=True
            If True, overwrite the file if it already exists. If False, raise
            an `IOError` if the file exists.
        audio_codec : str, optional
            ffmpeg encoder to use (e.g., ``"libmp3lame"``, ``"libopus"``,
            ``"aac"``). If not provided, ffmpeg picks the default codec for the
            output format.
        audio_bitrate : str, optional
            Target audio bitrate (e.g., ``"128k"``, ``"192k"``, ``"320k"``).
        audio_quality : str, optional
            Quality level for VBR encoding. Interpretation depends on the codec
            (e.g., ``"2"`` for libmp3lame VBR, ``"0"`` for best quality).
        ffmpeg_extra_args : list of str, optional
            Additional ffmpeg output arguments passed directly to the ffmpeg
            command line (e.g., ``["-cutoff", "20000"]``).

        Returns
        -------
        file : str
            The output filename with placeholders filled in.

        Raises
        ------
        IOError
            If `filename` exists and `exists_ok` is False.

        Examples
        --------
        Create an AudioRegion, specifying `start`. The `end` will be computed
        based on `start` and the region's duration.

        >>> region = AudioRegion(b'\0' * 2 * 24000,
        >>>                      sampling_rate=16000,
        >>>                      sample_width=2,
        >>>                      channels=1,
        >>>                      start=2.25)
        >>> region
        <AudioRegion(duration=1.500, sampling_rate=16000, sample_width=2, channels=1)>

        >>> assert region.end == 3.75
        >>> assert region.save('audio_{start}-{end}.wav') == "audio_2.25-3.75.wav"
        >>> filename = region.save('audio_{start:.3f}-{end:.3f}_{duration:.3f}.wav')
        >>> assert filename == "audio_2.250-3.750_1.500.wav"

        Save as MP3 with a specific bitrate:

        >>> region.save("output.mp3", audio_bitrate="192k")

        Save as OGG with a specific codec:

        >>> region.save("output.ogg", audio_codec="libopus", audio_bitrate="64k")
        """

        if isinstance(filename, Path):
            if not exists_ok and filename.exists():
                raise FileExistsError(
                    "file '{filename}' exists".format(filename=str(filename))
                )
        if isinstance(filename, str):
            filename = filename.format(
                duration=self.duration,
                start=self.start,
                end=self.end,
            )
            if not exists_ok and os.path.exists(filename):
                raise FileExistsError(
                    "file '{filename}' exists".format(filename=filename)
                )
        to_file(
            self.data,
            filename,
            audio_format,
            sr=self.sr,
            sw=self.sw,
            ch=self.ch,
            **audio_parameters,
        )
        return filename

    def split(
        self,
        *,
        min_dur: float = 0.2,
        max_dur: float | None = 5,
        max_silence: float = 0.3,
        max_leading_silence: float = 0,
        max_trailing_silence: float | None = None,
        drop_trailing_silence: bool | None = None,
        strict_min_dur: bool = False,
        **kwargs: Any,
    ) -> Generator[AudioRegion, None, None]:
        """
        Split audio region. See :func:`auditok.split` for a comprehensive
        description of split parameters.
        See Also :meth:`AudioRegio.split_and_plot`.
        """
        if kwargs.get("max_read", kwargs.get("mr")) is not None:
            warn_msg = "'max_read' (or 'mr') should not be used with "
            warn_msg += "AudioRegion.split_and_plot(). You should rather "
            warn_msg += "slice audio region before calling this method"
            raise RuntimeWarning(warn_msg)
        if drop_trailing_silence is not None:
            warnings.warn(
                _DROP_TRAILING_SILENCE_DEPRECATION,
                DeprecationWarning,
                stacklevel=2,
            )
            if max_trailing_silence is None:
                max_trailing_silence = 0.0 if drop_trailing_silence else None
        return split(
            self,
            min_dur=min_dur,
            max_dur=max_dur,
            max_silence=max_silence,
            max_leading_silence=max_leading_silence,
            max_trailing_silence=max_trailing_silence,
            strict_min_dur=strict_min_dur,
            **kwargs,
        )

    def trim(
        self,
        *,
        min_dur: float = 0.2,
        max_dur: float | None = 5,
        max_silence: float = 0.3,
        max_leading_silence: float = 0,
        max_trailing_silence: float | None = None,
        drop_trailing_silence: bool | None = None,
        strict_min_dur: bool = False,
        **kwargs: Any,
    ) -> AudioRegion:
        """
        Remove leading and trailing silence from this audio region.

        Detects audio activity using :meth:`split`, then returns the slice
        from the start of the first detection to the end of the last
        detection. All audio between detections (including internal silence)
        is preserved.

        See :func:`split` for parameter descriptions.

        Returns
        -------
        AudioRegion
            A new AudioRegion with leading and trailing silence removed.
            Returns an empty AudioRegion (zero duration) if no audio activity
            is detected, so the result can always be used for joining,
            concatenation, etc.

        See Also
        --------
        split : Split audio into individual activity regions.
        """
        if drop_trailing_silence is not None:
            warnings.warn(
                _DROP_TRAILING_SILENCE_DEPRECATION,
                DeprecationWarning,
                stacklevel=2,
            )
            if max_trailing_silence is None:
                max_trailing_silence = 0.0 if drop_trailing_silence else None
        first = None
        last = None
        for region in self.split(
            min_dur=min_dur,
            max_dur=max_dur,
            max_silence=max_silence,
            max_leading_silence=max_leading_silence,
            max_trailing_silence=max_trailing_silence,
            strict_min_dur=strict_min_dur,
            **kwargs,
        ):
            if first is None:
                first = region
            last = region
        if first is None:
            return AudioRegion(b"", self.sr, self.sw, self.ch)
        return self.sec[first.start : last.end]  # type: ignore[misc]

    def plot(
        self,
        scale_signal: bool = True,
        show: bool = True,
        figsize: tuple[float, float] | None = None,
        save_as: str | None = None,
        dpi: int = 120,
        theme: str | dict[str, Any] = "auditok",
    ) -> None:
        """
        Plot the audio region with one subplot per channel.

        Parameters
        ----------
        scale_signal : bool, optional, default=True
            If True, scale the signal by subtracting its mean and dividing by its
            standard deviation before plotting.
        show : bool, optional, default=False
            Whether to display the plot immediately after the function call.
        figsize : tuple, optional, default=None
            Width and height of the figure, passed to `matplotlib`.
        save_as : str, optional, default=None
            If specified, save the plot to the given filename.
        dpi : int, optional, default=120
            Dots per inch (DPI) for the plot, passed to `matplotlib`.
        theme : str or dict, optional, default="auditok"
            Plot theme to use. Only the "auditok" theme is currently implemented.
            To define a custom theme, refer to
            :attr:`auditok.plotting.AUDITOK_PLOT_THEME`.
        """

        plot(
            self,
            scale_signal=scale_signal,
            show=show,
            figsize=figsize,
            save_as=save_as,
            dpi=dpi,
            theme=theme,
        )

    def split_and_plot(
        self,
        *,
        min_dur: float = 0.2,
        max_dur: float | None = 5,
        max_silence: float = 0.3,
        max_leading_silence: float = 0,
        max_trailing_silence: float | None = None,
        drop_trailing_silence: bool | None = None,
        strict_min_dur: bool = False,
        scale_signal: bool = True,
        show: bool = True,
        figsize: tuple[float, float] | None = None,
        save_as: str | None = None,
        dpi: int = 120,
        theme: str | dict[str, Any] = "auditok",
        interactive: bool = False,
        **kwargs: Any,
    ) -> list[AudioRegion]:
        """
        Split the audio region, then plot the signal and detected regions.

        When ``interactive=True`` and running inside a Jupyter notebook, an
        HTML5 widget with a Canvas waveform is displayed instead of a
        matplotlib figure.  Clicking on a highlighted detection plays that
        event in the browser via the Web Audio API.  If not running in a
        notebook, the matplotlib plot is used as a fallback.

        Alias
        -----
        :meth:`splitp`

        Parameters
        ----------
        interactive : bool, optional, default=False
            If True and running inside a Jupyter notebook, display an
            interactive HTML/Canvas/WebAudio widget instead of a matplotlib
            plot.  Falls back to matplotlib when not in a notebook.

        Refer to :func:`auditok.split()` for a detailed description of split
        parameters, and to :meth:`plot` for plot-specific parameters.
        """
        if drop_trailing_silence is not None:
            warnings.warn(
                _DROP_TRAILING_SILENCE_DEPRECATION,
                DeprecationWarning,
                stacklevel=2,
            )
            if max_trailing_silence is None:
                max_trailing_silence = 0.0 if drop_trailing_silence else None
        regions: list[AudioRegion] = list(
            self.split(
                min_dur=min_dur,
                max_dur=max_dur,
                max_silence=max_silence,
                max_leading_silence=max_leading_silence,
                max_trailing_silence=max_trailing_silence,
                strict_min_dur=strict_min_dur,
                **kwargs,
            )
        )
        eth = kwargs.get(
            "energy_threshold", kwargs.get("eth", DEFAULT_ENERGY_THRESHOLD)
        )

        if interactive:
            from .widget import _in_notebook, display_interactive

            if _in_notebook():
                display_interactive(
                    self,
                    regions,
                    energy_threshold=eth,
                )
                return regions

        detections = ((reg.start, reg.end) for reg in regions)
        plot(
            self,
            scale_signal=scale_signal,
            detections=detections,
            energy_threshold=eth,
            show=show,
            figsize=figsize,
            save_as=save_as,
            dpi=dpi,
            theme=theme,
        )
        return regions

    def _check_other_parameters(self, other):
        if other.sr != self.sr:
            raise AudioParameterError(
                "Can only concatenate AudioRegions of the same "
                "sampling rate ({} != {})".format(self.sr, other.sr)
            )
        if other.sw != self.sw:
            raise AudioParameterError(
                "Can only concatenate AudioRegions of the same "
                "sample width ({} != {})".format(self.sw, other.sw)
            )
        if other.ch != self.ch:
            raise AudioParameterError(
                "Can only concatenate AudioRegions of the same "
                "number of channels ({} != {})".format(self.ch, other.ch)
            )

    def _check_iter_others(self, others):
        for other in others:
            self._check_other_parameters(other)
            yield other

    def join(self, others: Iterable[AudioRegion]) -> AudioRegion:
        data = self.data.join(
            other.data for other in self._check_iter_others(others)
        )
        return AudioRegion(data, self.sr, self.sw, self.ch)

    def __array__(self, dtype=None, copy=None):
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if copy:
            arr = arr.copy()
        return arr

    def numpy(self) -> np.ndarray:
        """Audio region a 2D numpy array of shape (n_channels, n_samples)."""
        return signal.to_array(self.data, self.sample_width, self.channels)

    def __len__(self) -> int:
        """
        Return region length in number of samples.
        """
        return len(self.data) // (self.sample_width * self.channels)

    @property
    def len(self) -> int:
        """
        Return the length of the audio region in number of samples.
        """

        return len(self)

    def __bytes__(self) -> bytes:
        return self.data

    def __str__(self) -> str:
        return (
            "AudioRegion(duration={:.3f}, "
            "sampling_rate={}, sample_width={}, channels={})".format(
                self.duration, self.sr, self.sw, self.ch
            )
        )

    def __repr__(self) -> str:
        return "<{}>".format(str(self))

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks.

        Returns an HTML5 audio player with the audio data embedded as a
        base64-encoded WAV, letting users play detected audio events directly
        in the browser.
        """
        import base64
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setframerate(self.sr)
            wf.setsampwidth(self.sw)
            wf.setnchannels(self.ch)
            wf.writeframes(self.data)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        src = f"data:audio/wav;base64,{b64}"

        if self.start is not None:
            time_info = " | {:.3f}s — {:.3f}s".format(self.start, self.end)
        else:
            time_info = ""
        label = (
            "<small>"
            "<b>AudioRegion</b> "
            "{:.3f}s{} "
            "({} Hz, {}‑bit, {} ch)"
            "</small>".format(
                self.duration,
                time_info,
                self.sr,
                self.sw * 8,
                self.ch,
            )
        )
        return (
            '<div style="margin:4px 0">'
            "{label}<br>"
            '<audio controls preload="auto" style="margin-top:2px">'
            '<source src="{src}" type="audio/wav">'
            "</audio>"
            "</div>"
        ).format(label=label, src=src)

    def __add__(self, other: AudioRegion) -> AudioRegion:
        """
        Concatenate this audio region with `other`, returning a new region.

        Both regions must have the same sampling rate, sample width, and number
        of channels. If they differ, a `ValueError` is raised.
        """

        if not isinstance(other, AudioRegion):
            raise TypeError(
                "Can only concatenate AudioRegion, "
                'not "{}"'.format(type(other))
            )
        self._check_other_parameters(other)
        data = self.data + other.data
        return AudioRegion(data, self.sr, self.sw, self.ch)

    def __radd__(self, other: AudioRegion | int) -> AudioRegion:
        """
        Concatenate `other` with this audio region.

        Parameters
        ----------
        other : AudioRegion or int
            An `AudioRegion` with the same audio parameters as this region, or
            `0` to enable concatenating multiple regions using `sum`.

        Returns
        -------
        AudioRegion
            A new `AudioRegion` representing the concatenation result.
        """
        if other == 0:
            return self
        return other.add(self)

    def __mul__(self, n: int) -> AudioRegion:
        if not isinstance(n, int):
            err_msg = "Can't multiply AudioRegion by a non-int of type '{}'"
            raise TypeError(err_msg.format(type(n)))
        data = self.data * n
        return AudioRegion(data, self.sr, self.sw, self.ch)

    def __rmul__(self, n: int) -> AudioRegion:
        return self * n

    def __truediv__(self, n: int) -> list[AudioRegion]:
        if not isinstance(n, int) or n <= 0:
            raise TypeError("AudioRegion can only be divided by a positive int")
        samples_per_sub_region, rest = divmod(len(self), n)
        onset = 0
        sub_regions = []
        while onset < len(self):
            offset = 0
            if rest > 0:
                offset = 1
                rest -= 1
            offset += onset + samples_per_sub_region
            sub_regions.append(self[onset:offset])
            onset = offset
        return sub_regions

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, AudioRegion):
            return False
        return (
            (self.data == other.data)
            and (self.sr == other.sr)
            and (self.sw == other.sw)
            and (self.ch == other.ch)
        )

    def __getitem__(self, index: slice) -> AudioRegion:
        err_msg = "Slicing AudioRegion by samples requires indices of type "
        err_msg += "'int' without a step (e.g. region.sec[1600:3200])"
        start_sample, stop_sample = _check_convert_index(index, (int), err_msg)

        bytes_per_sample = self.sample_width * self.channels
        len_samples = len(self.data) // bytes_per_sample

        if start_sample < 0:
            start_sample = max(start_sample + len_samples, 0)
        onset = start_sample * bytes_per_sample

        if stop_sample is not None:
            if stop_sample < 0:
                stop_sample = max(stop_sample + len_samples, 0)
            offset = index.stop * bytes_per_sample
        else:
            offset = None

        data = self.data[onset:offset]
        return AudioRegion(data, self.sr, self.sw, self.ch)


class StreamTokenizer:
    """
    Class for stream tokenizers, implementing a 4-state automaton scheme
    to extract relevant sub-sequences from a data stream in real-time.

    Parameters
    ----------
    validator : callable or :class:`DataValidator` (must implement `is_valid`).
        Called with each data frame read from the source. Should take a
        single argument and return True or False to indicate valid and
        invalid frames, respectively.

    min_length : int
        Minimum number of frames in a valid token, including any tolerated
        non-valid frames within the token.

    max_length : int or float("inf")
        Maximum number of frames in a valid token, including all tolerated
        non-valid frames within the token. Use float("inf") for no limit.

    max_continuous_silence : int
        Maximum number of consecutive non-valid frames within a token. Each
        silent region may contain up to `max_continuous_silence` frames.

    max_leading_silence : int, default=0
        Maximum number of non-valid frames to retain immediately before
        the first valid frame of each token. These frames are prepended to
        the token data, and the token's start position is adjusted backward
        accordingly.

        This is useful for audio events where the onset energy rises
        gradually (e.g., speech utterances whose first few frames fall below
        the energy threshold). Without leading silence, those frames are
        cut, producing an abrupt, perceptually uncomfortable start.
        Including a small amount of leading context (e.g., 3-6 analysis
        windows, ~150-300 ms) preserves the natural attack of the sound.

        When set to 0 (default), no leading silence is kept — the token
        starts at the first valid frame, preserving backward-compatible
        behavior.

    max_trailing_silence : int or None, default=None
        Maximum number of trailing non-valid frames to keep at the end of
        each token. When a token ends because ``max_continuous_silence`` is
        exceeded (or data ends), up to ``max_trailing_silence`` frames of
        the accumulated trailing silence are retained; the rest are trimmed.

        - ``None`` (default): keep all trailing silence (no trimming).
        - ``0``: drop all trailing silence (equivalent to the legacy
          ``DROP_TRAILING_SILENCE`` mode flag).
        - ``N > 0``: keep up to N trailing silent frames.

        This decouples the *perceptual padding* at the end of a token from
        ``max_continuous_silence``, which controls *when* a token ends.

    init_min : int, default=0
        Minimum number of consecutive valid frames required before
        tolerating any non-valid frames. Helps discard non-valid tokens
        early if needed.

        .. note::
            This parameter has never been exposed in the high-level
            :func:`split` API. For most use cases, validator already
            filters spurious frames, making ``init_min`` redundant.
            Consider using ``max_leading_silence`` instead, which addresses
            the more common need of preserving natural sound onsets.

    init_max_silence : int, default=0
        Maximum number of tolerated consecutive non-valid frames before
        reaching `init_min`. Used if `init_min` is specified.

        .. note::
            See ``init_min`` note above.

    mode : int
        Defines the tokenizer behavior with the following options:

        - `StreamTokenizer.NORMAL` (0, default): Do not drop trailing silence
          and allow tokens shorter than `min_length` if they immediately follow
          a delivered token.

        - `StreamTokenizer.STRICT_MIN_LENGTH` (2): If a token `i` is
          delivered at `max_length`, any adjacent token `i+1` must meet
          `min_length`.

        - `StreamTokenizer.DROP_TRAILING_SILENCE` (4): Drop all trailing
          non-valid frames from a token unless the token is truncated
          (e.g., at `max_length`).

        - `StreamTokenizer.STRICT_MIN_LENGTH | StreamTokenizer.DROP_TRAILING_SILENCE`:
          Apply both `STRICT_MIN_LENGTH` and `DROP_TRAILING_SILENCE`.

    Examples
    --------
    In the following, without `STRICT_MIN_LENGTH`, the 'BB' token is
    accepted even though it is shorter than `min_length` (3) because it
    immediately follows the last delivered token:

    >>> from auditok.core import StreamTokenizer
    >>> from auditok.util import StringDataSource, DataValidator

    >>> class UpperCaseChecker(DataValidator):
    >>>     def is_valid(self, frame):
                return frame.isupper()

    >>> dsource = StringDataSource("aaaAAAABBbbb")
    >>> tokenizer = StreamTokenizer(
    >>>     validator=UpperCaseChecker(),
    >>>     min_length=3,
    >>>     max_length=4,
    >>>     max_continuous_silence=0
    >>> )
    >>> tokenizer.tokenize(dsource)
    [(['A', 'A', 'A', 'A'], 3, 6), (['B', 'B'], 7, 8)]

    Using `STRICT_MIN_LENGTH` mode rejects the 'BB' token:

    >>> tokenizer = StreamTokenizer(
    >>>     validator=UpperCaseChecker(),
    >>>     min_length=3,
    >>>     max_length=4,
    >>>     max_continuous_silence=0,
    >>>     mode=StreamTokenizer.STRICT_MIN_LENGTH
    >>> )
    >>> tokenizer.tokenize(dsource)
    [(['A', 'A', 'A', 'A'], 3, 6)]

    With `DROP_TRAILING_SILENCE`, trailing silence is removed if not truncated:

    >>> tokenizer = StreamTokenizer(
    >>>     validator=UpperCaseChecker(),
    >>>     min_length=3,
    >>>     max_length=6,
    >>>     max_continuous_silence=3,
    >>>     mode=StreamTokenizer.DROP_TRAILING_SILENCE
    >>> )
    >>> dsource = StringDataSource("aaaAAAaaaBBbbbb")
    >>> tokenizer.tokenize(dsource)
    [(['A', 'A', 'A', 'a', 'a', 'a'], 3, 8), (['B', 'B'], 9, 10)]

    Without `DROP_TRAILING_SILENCE`, the output includes trailing frames:

    .. code:: python

        [
            (['A', 'A', 'A', 'a', 'a', 'a'], 3, 8),
            (['B', 'B', 'b', 'b', 'b'], 9, 13)
        ]
    """

    SILENCE = 0
    POSSIBLE_SILENCE = 1
    POSSIBLE_NOISE = 2
    NOISE = 3
    NORMAL = 0
    STRICT_MIN_LENGTH = 2
    DROP_TRAILING_SILENCE = 4

    def __init__(
        self,
        validator: DataValidator | Callable[[Any], bool],
        min_length: int,
        max_length: int | float,
        max_continuous_silence: int,
        init_min: int = 0,
        init_max_silence: int = 0,
        mode: int = 0,
        max_leading_silence: int = 0,
        max_trailing_silence: int | None = None,
    ) -> None:
        if callable(validator):
            self._is_valid = validator
        elif isinstance(validator, DataValidator):
            self._is_valid = validator.is_valid
        else:
            raise TypeError(
                "'validator' must be a callable or an instance of "
                "DataValidator"
            )

        if max_length <= 0:
            raise ValueError(
                "'max_length' must be > 0 (value={0})".format(max_length)
            )

        if min_length <= 0 or min_length > max_length:
            err_msg = "'min_length' must be > 0 and <= 'max_length' (value={0})"
            raise ValueError(err_msg.format(min_length))

        if max_continuous_silence >= max_length:
            err_msg = "'max_continuous_silence' must be < 'max_length' "
            err_msg += "(value={0})"
            raise ValueError(err_msg.format(max_continuous_silence))

        if init_min >= max_length:
            raise ValueError(
                "'init_min' must be < 'max_length' (value={0})".format(
                    max_continuous_silence
                )
            )

        self.validator = validator
        self.min_length = min_length
        self.max_length = max_length
        self.max_continuous_silence = max_continuous_silence
        self.init_min = init_min
        self.init_max_silent = init_max_silence
        self.max_leading_silence = max_leading_silence
        self._set_mode(mode)
        # max_trailing_silence takes precedence over DROP_TRAILING_SILENCE flag
        if max_trailing_silence is not None:
            self.max_trailing_silence = max_trailing_silence
        elif self._drop_trailing_silence:
            self.max_trailing_silence = 0
        else:
            self.max_trailing_silence = None
        self._deliver = None
        self._tokens = None
        self._state = None
        self._data = None
        self._contiguous_token = False
        self._init_count = 0
        self._silence_length = 0
        self._start_frame = 0
        self._current_frame = 0
        self._leading_buffer: deque[Any] = deque(maxlen=max_leading_silence)

    def _set_mode(self, mode):
        strict_min_and_drop_trailing = StreamTokenizer.STRICT_MIN_LENGTH
        strict_min_and_drop_trailing |= StreamTokenizer.DROP_TRAILING_SILENCE
        if mode not in [
            StreamTokenizer.NORMAL,
            StreamTokenizer.STRICT_MIN_LENGTH,
            StreamTokenizer.DROP_TRAILING_SILENCE,
            strict_min_and_drop_trailing,
        ]:
            raise ValueError("Wrong value for mode")
        self._mode = mode
        self._strict_min_length = (mode & self.STRICT_MIN_LENGTH) != 0
        self._drop_trailing_silence = (mode & self.DROP_TRAILING_SILENCE) != 0

    def _reinitialize(self):
        self._contiguous_token = False
        self._data = []
        self._tokens = []
        self._state = self.SILENCE
        self._current_frame = -1
        self._deliver = self._append_token
        self._leading_buffer.clear()

    def tokenize(
        self,
        data_source: DataSource,
        callback: Callable[..., Any] | None = None,
        generator: bool = False,
    ) -> list[tuple[list[Any], int, int]] | Generator | None:
        """
        Read data from `data_source` one frame at a time and process each frame
        to detect sequences that form valid tokens.

        Parameters
        ----------
        data_source : DataSource
            An instance of the :class:`DataSource` class that implements a `read`
            method. `read` should return a slice of the signal (a frame of any
            type that can be processed by the validator) or None when there is no
            more data in the source.

        callback : callable, optional
            A function that takes three arguments. If provided, `callback` is
            called each time a valid token is detected.

        generator : bool, optional, default=False
            If True, the method yields tokens as they are detected, rather than
            returning a list. If False, a list of tokens is returned.

        Returns
        -------
        list of tuples or generator
            A list of tokens if `generator` is False, or a generator yielding
            tokens if `generator` is True. Each token is a tuple with the
            following structure:

            .. code:: python

                (data, start, end)

            where `data` is a list of frames in the token, `start` is the index
            of the first frame in the original data, and `end` is the index of
            the last frame.
        """

        token_gen = self._iter_tokens(data_source)
        if callback:
            for token in token_gen:
                callback(*token)
            return None
        if generator:
            return token_gen
        return list(token_gen)

    def _iter_tokens(self, data_source):
        self._reinitialize()
        while True:
            frame = data_source.read()
            self._current_frame += 1
            if frame is None:
                token = self._post_process()
                if token is not None:
                    yield token
                break
            token = self._process(frame)
            if token is not None:
                yield token

    def _process(self, frame):  # noqa: C901

        frame_is_valid = self._is_valid(frame)

        if self._state == self.SILENCE:

            if frame_is_valid:
                # seems we got a valid frame after a silence
                leading = list(self._leading_buffer)
                self._leading_buffer.clear()
                self._init_count = 1
                self._silence_length = 0
                self._start_frame = self._current_frame - len(leading)
                self._data = leading + [frame]

                if self._init_count >= self.init_min:
                    self._state = self.NOISE
                    if len(self._data) >= self.max_length:
                        return self._process_end_of_detection(True)
                else:
                    self._state = self.POSSIBLE_NOISE
            else:
                self._leading_buffer.append(frame)

        elif self._state == self.POSSIBLE_NOISE:

            if frame_is_valid:
                self._silence_length = 0
                self._init_count += 1
                self._data.append(frame)
                if self._init_count >= self.init_min:
                    self._state = self.NOISE
                    if len(self._data) >= self.max_length:
                        return self._process_end_of_detection(True)

            else:
                self._silence_length += 1
                if (
                    self._silence_length > self.init_max_silent
                    or len(self._data) + 1 >= self.max_length
                ):
                    # either init_max_silent or max_length is reached
                    # before _init_count, back to silence
                    self._data = []
                    self._state = self.SILENCE
                else:
                    self._data.append(frame)

        elif self._state == self.NOISE:

            if frame_is_valid:
                self._data.append(frame)
                if len(self._data) >= self.max_length:
                    return self._process_end_of_detection(True)

            elif self.max_continuous_silence <= 0:
                # max token reached at this frame will _deliver if
                # _contiguous_token and not _strict_min_length
                self._state = self.SILENCE
                return self._process_end_of_detection()
            else:
                # this is the first silent frame following a valid one
                # and it is tolerated
                self._silence_length = 1
                self._data.append(frame)
                self._state = self.POSSIBLE_SILENCE
                if len(self._data) == self.max_length:
                    return self._process_end_of_detection(True)
                    # don't reset _silence_length because we still
                    # need to know the total number of silent frames

        elif self._state == self.POSSIBLE_SILENCE:

            if frame_is_valid:
                self._data.append(frame)
                self._silence_length = 0
                self._state = self.NOISE
                if len(self._data) >= self.max_length:
                    return self._process_end_of_detection(True)

            else:
                if self._silence_length >= self.max_continuous_silence:
                    self._state = self.SILENCE
                    if self._silence_length < len(self._data):
                        # _deliver only gathered frames aren't all silent
                        return self._process_end_of_detection()
                    self._data = []
                    self._silence_length = 0
                else:
                    self._data.append(frame)
                    self._silence_length += 1
                    if len(self._data) >= self.max_length:
                        return self._process_end_of_detection(True)
                        # don't reset _silence_length because we still
                        # need to know the total number of silent frames

    def _post_process(self):
        if self._state == self.NOISE or self._state == self.POSSIBLE_SILENCE:
            if len(self._data) > 0 and len(self._data) > self._silence_length:
                return self._process_end_of_detection()

    def _process_end_of_detection(self, truncated=False):

        if (
            not truncated
            and self.max_trailing_silence is not None
            and self._silence_length > self.max_trailing_silence
        ):
            # Trim trailing silence beyond the allowed amount.
            # happens if max_continuous_silence is reached
            # or max_length is reached at a silent frame
            excess = self._silence_length - self.max_trailing_silence
            self._data = self._data[:-excess]

        if (len(self._data) >= self.min_length) or (
            len(self._data) > 0
            and not self._strict_min_length
            and self._contiguous_token
        ):

            start_frame = self._start_frame
            end_frame = self._start_frame + len(self._data) - 1
            data = self._data
            self._data = []
            token = (data, start_frame, end_frame)

            if truncated:
                # next token (if any) will start at _current_frame + 1
                self._start_frame = self._current_frame + 1
                # remember that it is contiguous with the just delivered one
                self._contiguous_token = True
            else:
                self._contiguous_token = False
            return token
        else:
            self._contiguous_token = False

        self._data = []

    def _append_token(self, data, start, end):
        self._tokens.append((data, start, end))
