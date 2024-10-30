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

import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import AudioParameterError, TooSmallBlockDuration
from .io import check_audio_data, get_audio_source, player_for, to_file
from .plotting import plot
from .util import AudioEnergyValidator, AudioReader, DataValidator

try:
    from . import signal_numpy as signal
except ImportError:
    from . import signal

__all__ = [
    "load",
    "split",
    "make_silence",
    "split_and_join_with_silence",
    "AudioRegion",
    "StreamTokenizer",
]


DEFAULT_ANALYSIS_WINDOW = 0.05
DEFAULT_ENERGY_THRESHOLD = 50
_EPSILON = 1e-10


def load(input, skip=0, max_read=None, **kwargs):
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
        microphone using PyAudio. For `bytes` data or a raw audio file path,
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
    input,
    min_dur=0.2,
    max_dur=5,
    max_silence=0.3,
    drop_trailing_silence=False,
    strict_min_dur=False,
    **kwargs,
):
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
        specify explicitly via `audio_format` or `fmt`. Otherwise, the backend
        (currently only `pydub`) handles loading data.

    min_dur : float, default=0.2
        Minimum duration in seconds of a detected audio event. Higher values
        can exclude very short utterances (e.g., single words like "yes" or
        "no"). Lower values may increase the number of short audio events.

    max_dur : float, default=5
        Maximum duration in seconds for an audio event. Events longer than this
        are truncated. If the remainder of a truncated event is shorter than
        `min_dur`, it is included as a valid event if `strict_min_dur` is False;
        otherwise, it is rejected.

    max_silence : float, default=0.3
        Maximum duration of continuous silence allowed within an audio event.
        Multiple silent gaps of this duration may appear in a single event.
        Trailing silence at the end of an event is kept if
        `drop_trailing_silence` is False.

    drop_trailing_silence : bool, default=False
        Whether to remove trailing silence from detected events. To prevent
        abrupt speech cuts, it is recommended to keep trailing silence, so
        default is False.

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
    if max_dur <= 0:
        raise ValueError(f"'max_dur' ({max_dur}) must be > 0")
    if max_silence < 0:
        raise ValueError(f"'max_silence' ({max_silence}) must be >= 0")

    if isinstance(input, AudioReader):
        source = input
        analysis_window = source.block_dur
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

    validator = kwargs.get("validator", kwargs.get("val"))
    if validator is None:
        energy_threshold = kwargs.get(
            "energy_threshold", kwargs.get("eth", DEFAULT_ENERGY_THRESHOLD)
        )
        use_channel = kwargs.get("use_channel", kwargs.get("uc"))
        validator = AudioEnergyValidator(
            energy_threshold, source.sw, source.ch, use_channel=use_channel
        )
    mode = StreamTokenizer.DROP_TRAILING_SILENCE if drop_trailing_silence else 0
    if strict_min_dur:
        mode |= StreamTokenizer.STRICT_MIN_LENGTH
    min_length = _duration_to_nb_windows(min_dur, analysis_window, math.ceil)
    max_length = _duration_to_nb_windows(
        max_dur, analysis_window, math.floor, _EPSILON
    )
    max_continuous_silence = _duration_to_nb_windows(
        max_silence, analysis_window, math.floor, _EPSILON
    )

    err_msg = "({0} sec.) results in {1} analysis window(s) "
    err_msg += "({1} == {6}({0} / {2})) which is {5} the number "
    err_msg += "of analysis window(s) for 'max_dur' ({3} == floor({4} / {2}))"
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
        validator, min_length, max_length, max_continuous_silence, mode=mode
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


def make_silence(duration, sampling_rate=16000, sample_width=2, channels=1):
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


def split_and_join_with_silence(input, silence_duration, **kwargs):
    """
    Split input audio and join (glue) the resulting regions with a specified
    silence duration between them. This can be used to adjust the length of
    silence between audio events, either shortening or lengthening pauses.

    Parameters
    ----------
    silence_duration : float
        Duration of silence in seconds between audio events.

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

    def __init__(self, region):
        self._region = region

    def __getitem__(self, index):
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


class _AudioRegionMetadata(dict):
    """A class to store :class:`AudioRegion`'s metadata."""

    def __getattr__(self, name):
        warnings.warn(
            "`AudioRegion.meta` is deprecated and will be removed in future "
            "versions. For the 'start' and 'end' fields, please use "
            "`AudioRegion.start` and `AudioRegion.end`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name in self:
            return self[name]
        else:
            err_msg = "AudioRegion metadata has no entry '{}'"
            raise AttributeError(err_msg.format(name))

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        return "\n".join("{}: {}".format(k, v) for k, v in self.items())

    def __repr__(self):
        return str(self)


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
    start: float = field(default=None, repr=None)

    def __post_init__(self):

        check_audio_data(self.data, self.sample_width, self.channels)
        object.__setattr__(self, "splitp", self.split_and_plot)
        duration = len(self.data) / (
            self.sampling_rate * self.sample_width * self.channels
        )
        object.__setattr__(self, "duration", duration)

        if self.start is not None:
            object.__setattr__(self, "end", self.start + self.duration)
            object.__setattr__(
                self,
                "meta",
                _AudioRegionMetadata({"start": self.start, "end": self.end}),
            )
        else:
            object.__setattr__(self, "end", None)
            object.__setattr__(self, "meta", None)

        # `seconds` and `millis` are defined below as @property with docstring
        object.__setattr__(self, "_seconds_view", _SecondsView(self))
        object.__setattr__(self, "_millis_view", _MillisView(self))

        object.__setattr__(self, "sec", self.seconds)
        object.__setattr__(self, "s", self.seconds)
        object.__setattr__(self, "ms", self.millis)

    @classmethod
    def load(cls, input, skip=0, max_read=None, **kwargs):
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
    def seconds(self):
        """
        A view to slice audio region by seconds using
        ``region.seconds[start:end]``.
        """
        return self._seconds_view

    @property
    def millis(self):
        """A view to slice audio region by milliseconds using
        ``region.millis[start:end]``."""
        return self._millis_view

    @property
    def sr(self):
        """Sampling rate of audio data, alias for `sampling_rate`."""
        return self.sampling_rate

    @property
    def sw(self):
        """Number of bytes per sample, alias for `sample_width`."""
        return self.sample_width

    @property
    def ch(self):
        """Number of channels of audio data, alias for `channels`."""
        return self.channels

    def play(self, progress_bar=False, player=None, **progress_bar_kwargs):
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
        self, filename, audio_format=None, exists_ok=True, **audio_parameters
    ):
        """
        Save the audio region to a file.

        Parameters
        ----------
        filename : str or Path
            Path to the output audio file. If a string, it may include `{start}`,
            `{end}`, and `{duration}` placeholders. Regions created by `split`
            contain `start` and `end` attributes that can be used to format the
            filename, as shown in the example.
        audio_format : str, optional, default=None
            Format used to save the audio data. If None (default), the format is
            inferred from the file extension. If the filename has no extension,
            the audio is saved as a raw (headerless) audio file.
        exists_ok : bool, optional, default=True
            If True, overwrite the file if it already exists. If False, raise an
            `IOError` if the file exists.
        audio_parameters : dict, optional
            Additional keyword arguments to pass to the audio-saving backend.

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
        """

        if isinstance(filename, Path):
            if not exists_ok and filename.exists():
                raise FileExistsError(
                    "file '{filename}' exists".format(filename=str(filename))
                )
        if isinstance(filename, str):
            filename = filename.format(
                duration=self.duration,
                meta=self.meta,
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
            audio_parameters=audio_parameters,
        )
        return filename

    def split(
        self,
        min_dur=0.2,
        max_dur=5,
        max_silence=0.3,
        drop_trailing_silence=False,
        strict_min_dur=False,
        **kwargs,
    ):
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
        return split(
            self,
            min_dur=min_dur,
            max_dur=max_dur,
            max_silence=max_silence,
            drop_trailing_silence=drop_trailing_silence,
            strict_min_dur=strict_min_dur,
            **kwargs,
        )

    def plot(
        self,
        scale_signal=True,
        show=True,
        figsize=None,
        save_as=None,
        dpi=120,
        theme="auditok",
    ):
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
        min_dur=0.2,
        max_dur=5,
        max_silence=0.3,
        drop_trailing_silence=False,
        strict_min_dur=False,
        scale_signal=True,
        show=True,
        figsize=None,
        save_as=None,
        dpi=120,
        theme="auditok",
        **kwargs,
    ):
        """
        Split the audio region, then plot the signal and detected regions.

        Alias
        -----
        :meth:`splitp`

        Refer to :func:`auditok.split()` for a detailed description of split
        parameters, and to :meth:`plot` for plot-specific parameters.
        """
        regions = self.split(
            min_dur=min_dur,
            max_dur=max_dur,
            max_silence=max_silence,
            drop_trailing_silence=drop_trailing_silence,
            strict_min_dur=strict_min_dur,
            **kwargs,
        )
        regions = list(regions)
        detections = ((reg.meta.start, reg.meta.end) for reg in regions)
        eth = kwargs.get(
            "energy_threshold", kwargs.get("eth", DEFAULT_ENERGY_THRESHOLD)
        )
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

    def join(self, others):
        data = self.data.join(
            other.data for other in self._check_iter_others(others)
        )
        return AudioRegion(data, self.sr, self.sw, self.ch)

    @property
    def samples(self):
        warnings.warn(
            "`AudioRegion.samples` is deprecated and will be removed in future "
            "versions. Please use `AudioRegion.numpy()`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.numpy()

    def __array__(self):
        return self.numpy()

    def numpy(self):
        """Audio region a 2D numpy array of shape (n_channels, n_samples)."""
        return signal.to_array(self.data, self.sample_width, self.channels)

    def __len__(self):
        """
        Return region length in number of samples.
        """
        return len(self.data) // (self.sample_width * self.channels)

    @property
    def len(self):
        """
        Return the length of the audio region in number of samples.
        """

        return len(self)

    def __bytes__(self):
        return self.data

    def __str__(self):
        return (
            "AudioRegion(duration={:.3f}, "
            "sampling_rate={}, sample_width={}, channels={})".format(
                self.duration, self.sr, self.sw, self.ch
            )
        )

    def __repr__(self):
        return "<{}>".format(str(self))

    def __add__(self, other):
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

    def __radd__(self, other):
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

    def __mul__(self, n):
        if not isinstance(n, int):
            err_msg = "Can't multiply AudioRegion by a non-int of type '{}'"
            raise TypeError(err_msg.format(type(n)))
        data = self.data * n
        return AudioRegion(data, self.sr, self.sw, self.ch)

    def __rmul__(self, n):
        return self * n

    def __truediv__(self, n):
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

    def __eq__(self, other):
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

    def __getitem__(self, index):
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

    max_length : int
        Maximum number of frames in a valid token, including all tolerated
        non-valid frames within the token.

    max_continuous_silence : int
        Maximum number of consecutive non-valid frames within a token. Each
        silent region may contain up to `max_continuous_silence` frames.

    init_min : int, default=0
        Minimum number of consecutive valid frames required before
        tolerating any non-valid frames. Helps discard non-valid tokens
        early if needed.

    init_max_silence : int, default=0
        Maximum number of tolerated consecutive non-valid frames before
        reaching `init_min`. Used if `init_min` is specified.

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
        validator,
        min_length,
        max_length,
        max_continuous_silence,
        init_min=0,
        init_max_silence=0,
        mode=0,
    ):
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
        self._set_mode(mode)
        self._deliver = None
        self._tokens = None
        self._state = None
        self._data = None
        self._contiguous_token = False
        self._init_count = 0
        self._silence_length = 0
        self._start_frame = 0
        self._current_frame = 0

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

    def tokenize(self, data_source, callback=None, generator=False):
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
            return
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
                self._init_count = 1
                self._silence_length = 0
                self._start_frame = self._current_frame
                self._data.append(frame)

                if self._init_count >= self.init_min:
                    self._state = self.NOISE
                    if len(self._data) >= self.max_length:
                        return self._process_end_of_detection(True)
                else:
                    self._state = self.POSSIBLE_NOISE

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
            and self._drop_trailing_silence
            and self._silence_length > 0
        ):
            # happens if max_continuous_silence is reached
            # or max_length is reached at a silent frame
            self._data = self._data[0 : -self._silence_length]

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
