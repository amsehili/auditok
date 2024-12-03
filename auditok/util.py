"""
Module for high-level audio input-output operations.

.. autosummary::
    :toctree: generated/

    AudioEnergyValidator
    AudioReader
    Recorder
    make_duration_formatter
    make_channel_selector
"""

from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from . import signal
from .exceptions import TimeFormatError, TooSmallBlockDuration
from .io import AudioIOError, AudioSource, BufferAudioSource, get_audio_source

__all__ = [
    "make_duration_formatter",
    "make_channel_selector",
    "DataSource",
    "DataValidator",
    "StringDataSource",
    "AudioReader",
    "Recorder",
    "AudioEnergyValidator",
]


def make_duration_formatter(fmt):
    """
    Create and return a function to format durations in seconds using a
    specified format. Accepted format directives are:

    - ``%S`` : absolute seconds with 3 decimals; must be used alone.
    - ``%i`` : milliseconds
    - ``%s`` : seconds
    - ``%m`` : minutes
    - ``%h`` : hours

    The last four directives (%i, %s, %m, %h) should all be specified and can
    be placed in any order within the input format string.

    Parameters
    ----------
    fmt : str
        Format string specifying the duration format.

    Returns
    -------
    formatter : callable
        A function that takes a duration in seconds (float) and returns a
        formatted string.

    Raises
    ------
    TimeFormatError
        Raised if the format contains an unknown directive.

    Examples
    --------
    Using ``%S`` for total seconds with three decimal precision:

    .. code:: python

        formatter = make_duration_formatter("%S")
        formatter(123.589)
        # '123.589'
        formatter(123)
        # '123.000'

    Using combined directives:

    .. code:: python

        formatter = make_duration_formatter("%h:%m:%s.%i")
        formatter(3723.25)
        # '01:02:03.250'

        formatter = make_duration_formatter("%h hrs, %m min, %s sec and %i ms")
        formatter(3723.25)
        # '01 hrs, 02 min, 03 sec and 250 ms'

    Note:
        Omitting any of the four main directives (%i, %s, %m, %h) may result
        in incorrect formatting:

        .. code:: python

            formatter = make_duration_formatter("%m min, %s sec and %i ms")
            formatter(3723.25)
            # '02 min, 03 sec and 250 ms'
    """

    if fmt == "%S":

        def formatter(seconds):
            return "{:.3f}".format(seconds)

    elif fmt == "%I":

        def formatter(seconds):
            return "{0}".format(int(seconds * 1000))

    else:
        fmt = fmt.replace("%h", "{hrs:02d}")
        fmt = fmt.replace("%m", "{mins:02d}")
        fmt = fmt.replace("%s", "{secs:02d}")
        fmt = fmt.replace("%i", "{millis:03d}")
        try:
            i = fmt.index("%")
            raise TimeFormatError(
                "Unknown time format directive '{0}'".format(fmt[i : i + 2])
            )
        except ValueError:
            pass

        def formatter(seconds):
            millis = int(seconds * 1000)
            hrs, millis = divmod(millis, 3600000)
            mins, millis = divmod(millis, 60000)
            secs, millis = divmod(millis, 1000)
            return fmt.format(hrs=hrs, mins=mins, secs=secs, millis=millis)

    return formatter


def make_channel_selector(sample_width, channels, selected=None):
    """
    Create and return a callable for selecting a specific audio channel. The
    returned `selector` function can be used as `selector(audio_data)` and
    returns data for the specified channel.

    If `selected` is None or "any", the `selector` will separate and return a
    list of available channels: `[data_channel_1, data_channel_2, ...]`.

    Note that `selector` expects input data in `bytes` format but does not
    necessarily return a `bytes` object. To select or compute the desired
    channel (or average channel if `selected="avg"`), it converts the input
    data into an `array.array` or `numpy.ndarray`. After selection, the data
    is returned as is, without reconversion to `bytes`, for efficiency. The
    output can be converted back to `bytes` with `bytes(obj)` if needed.

    Special case: If `channels=1`, the input data is returned without processing.

    Parameters
    ----------
    sample_width : int
        Number of bytes per audio sample; should be 1, 2, or 4.
    channels : int
        Number of channels in the audio data that the selector should expect.
    selected : int or str, optional
        Channel to select in each call to `selector(raw_data)`. Acceptable values:
        - An integer in range [-channels, channels).
        - "mix", "avg", or "average" for averaging across channels.
        - None or "any" to return a list of all channels.

    Returns
    -------
    selector : callable
        A function that can be called as `selector(audio_data)` and returns data
        for the selected channel.

    Raises
    ------
    ValueError
        If `sample_width` is not one of {1, 2, 4}, or if `selected` has an
        unsupported value.
    """
    to_array_ = partial(
        signal.to_array, sample_width=sample_width, channels=channels
    )
    if channels == 1 or selected in (None, "any"):
        return to_array_

    if isinstance(selected, int):
        if selected < 0:
            selected += channels
        if selected < 0 or selected >= channels:
            err_msg = "Selected channel must be >= -channels and < channels"
            err_msg += ", given: {}"
            raise ValueError(err_msg.format(selected))
        return lambda x: to_array_(x)[selected]

    if selected in ("mix", "avg", "average"):
        return lambda x: to_array_(x).mean(axis=0)

    raise ValueError(
        "Selected channel must be an integer, None (alias 'any') or 'average' "
        "(alias 'avg' or 'mix')"
    )


class DataSource(ABC):
    """
    Base class for objects used as data sources in
    :func:`StreamTokenizer.tokenize`.

    Subclasses should implement a :func:`DataSource.read` method, which is
    expected to return a frame (or slice) of data from the source, and None
    when there is no more data to read.
    """

    @abstractmethod
    def read(self):
        """
        Read a block (or window) of data from this source.

        Returns
        -------
        data : object or None
            A block of data from the source. If no more data is available,
            should return None.
        """


class DataValidator(ABC):
    """
    Base class for validator objects used by :class:`.core.StreamTokenizer`
    to verify the validity of read data.

    Subclasses should implement the :func:`is_valid` method to define the
    specific criteria for data validity.
    """

    @abstractmethod
    def is_valid(self, data):
        """
        Determine whether the provided `data` meets validity criteria.

        Parameters
        ----------
        data : object
            The data to be validated.

        Returns
        -------
        bool
            True if `data` is valid, otherwise False.
        """


class AudioEnergyValidator(DataValidator):
    """
    A validator based on audio signal energy. For an input window of `N` audio
    samples (see :func:`AudioEnergyValidator.is_valid`), the energy is computed
    as:

    .. math::
        \\text{energy} = 20 \\log(\\sqrt({1}/{N} \\sum_{i=1}^{N} {a_i}^2))  % # noqa: W605

    where `a_i` represents the i-th audio sample.

    Parameters
    ----------
    energy_threshold : float
        Minimum energy required for an audio window to be considered valid.
    sample_width : int
        Size in bytes of a single audio sample.
    channels : int
        Number of audio channels in the data.
    use_channel : {None, "any", "mix", "avg", "average"} or int
        Specifies the channel used for energy computation:

        - None or "any": Compute energy for each channel and return the maximum.
        - "mix" (or "avg" / "average"): Average across all channels, then
          compute energy.
        - int (0 <= value < `channels`): Compute energy for the specified channel
          only, ignoring others.

    Returns
    -------
    energy : float
        Computed energy of the audio window, used to validate if the window
        meets the `energy_threshold`.
    """

    def __init__(
        self, energy_threshold, sample_width, channels, use_channel=None
    ):
        self._energy_threshold = energy_threshold
        self._sample_width = sample_width
        self._selector = make_channel_selector(
            sample_width, channels, use_channel
        )
        self._energy_agg_fn = np.max if use_channel in (None, "any") else None

    def is_valid(self, data):
        """
        Determine if the audio data meets the energy threshold.

        Parameters
        ----------
        data : bytes-like
            An array of raw audio data.

        Returns
        -------
        bool
            True if the energy of the audio data is greater than or equal to
            the specified threshold; otherwise, False.
        """

        log_energy = signal.calculate_energy(
            self._selector(data), self._energy_agg_fn
        )
        return log_energy >= self._energy_threshold


class StringDataSource(DataSource):
    """
    A :class:`DataSource` implementation that reads from a string buffer.

    Each call to :meth:`read` returns one character from the buffer and advances
    by one position. When the end of the buffer is reached, :meth:`read` returns
    None.

    Parameters
    ----------
    data : str
        The string data to be used as the source.
    """

    def __init__(self, data):

        self._data = None
        self._current = 0
        self.set_data(data)

    def read(self):
        """
        Read one character from buffer.

        Returns
        -------
        char : str
            current character or None if the end of the buffer is reached.
        """

        if self._current >= len(self._data):
            return None
        self._current += 1
        return self._data[self._current - 1]

    def set_data(self, data):
        """
        Set a new data buffer.

        Parameters
        ----------
        data : str
            new data buffer.
        """

        if not isinstance(data, str):
            raise ValueError("data must an instance of str")
        self._data = data
        self._current = 0


class _AudioReadingProxy:
    def __init__(self, audio_source):

        self._audio_source = audio_source

    def rewind(self):
        if self.rewindable:
            self._audio_source.rewind()
        else:
            raise AudioIOError("Audio stream is not rewindable")

    def rewindable(self):
        try:
            return self._audio_source.rewindable
        except AttributeError:
            return False

    def is_open(self):
        return self._audio_source.is_open()

    def open(self):
        self._audio_source.open()

    def close(self):
        self._audio_source.close()

    def read(self, size):
        return self._audio_source.read(size)

    @property
    def data(self):
        err_msg = "This AudioReader is not a recorder, no recorded data can "
        err_msg += "be retrieved"
        raise AttributeError(err_msg)

    def __getattr__(self, name):
        return getattr(self._audio_source, name)


class _Recorder(_AudioReadingProxy):
    """
    A class for `AudioReader` objects that records all data read from the source.

    This class is particularly useful for capturing audio data when reading from
    a microphone or similar live audio sources.
    """

    def __init__(self, audio_source):
        super().__init__(audio_source)
        self._cache = []
        self._read_block = self._read_and_cache
        self._read_from_cache = False
        self._data = None

    def read(self, size):
        return self._read_block(size)

    @property
    def data(self):
        if self._data is None:
            err_msg = "Un-rewinded recorder. `rewind` should be called before "
            err_msg += "accessing recorded data"
            raise RuntimeError(err_msg)
        return self._data

    def rewindable(self):
        return True

    def rewind(self):
        if self._read_from_cache:
            self._audio_source.rewind()
        else:
            self._data = b"".join(self._cache)
            self._cache = None
            self._audio_source = BufferAudioSource(
                self._data, self.sr, self.sw, self.ch
            )
            self._read_block = self._audio_source.read
            self.open()
            self._read_from_cache = True

    def _read_and_cache(self, size):
        # Read and save read data
        block = self._audio_source.read(size)
        if block is not None:
            self._cache.append(block)
        return block


class _Limiter(_AudioReadingProxy):
    """
    A class for `AudioReader` objects that restricts the amount of data read.

    This class is useful for limiting data intake when reading from a microphone
    or large audio files, ensuring only a specified amount of data is processed.
    """

    def __init__(self, audio_source, max_read):
        super().__init__(audio_source)
        self._max_read = max_read
        self._max_samples = round(max_read * self.sr)
        self._bytes_per_sample = self.sw * self.ch
        self._read_samples = 0

    @property
    def data(self):
        data = self._audio_source.data
        max_read_bytes = self._max_samples * self._bytes_per_sample
        return data[:max_read_bytes]

    @property
    def max_read(self):
        return self._max_read

    def read(self, size):
        size = min(self._max_samples - self._read_samples, size)
        if size <= 0:
            return None
        block = self._audio_source.read(size)
        if block is None:
            return None
        self._read_samples += len(block) // self._bytes_per_sample
        return block

    def rewind(self):
        super().rewind()
        self._read_samples = 0


class _FixedSizeAudioReader(_AudioReadingProxy):
    """
    A class to read fixed-size audio windows from a source.
    """

    def __init__(self, audio_source, block_dur):
        super().__init__(audio_source)

        if block_dur <= 0:
            raise ValueError(
                "block_dur must be > 0, given: {}".format(block_dur)
            )

        self._block_size = int(block_dur * self.sr)
        if self._block_size == 0:
            err_msg = "Too small block_dur ({0:f}) for sampling rate ({1}). "
            err_msg += "block_dur should cover at least one sample "
            err_msg += "(i.e. 1/{1})"
            raise TooSmallBlockDuration(
                err_msg.format(block_dur, self.sr), block_dur, self.sr
            )

    def read(self):
        return self._audio_source.read(self._block_size)

    @property
    def block_size(self):
        return self._block_size

    @property
    def block_dur(self):
        return self._block_size / self.sr

    def __getattr__(self, name):
        return getattr(self._audio_source, name)


class _OverlapAudioReader(_FixedSizeAudioReader):
    """
    A class for `AudioReader` objects that reads and returns overlapping audio
    windows.

    Useful for applications requiring overlapping segments, such as audio
    analysis or feature extraction.
    """

    def __init__(self, audio_source, block_dur, hop_dur):

        if hop_dur >= block_dur:
            raise ValueError('"hop_dur" should be <= "block_dur"')

        super().__init__(audio_source, block_dur)

        self._hop_size = int(hop_dur * self.sr)
        self._blocks = self._iter_blocks_with_overlap()

    def _iter_blocks_with_overlap(self):
        while not self.is_open():
            yield AudioIOError
        block = self._audio_source.read(self._block_size)
        if block is None:
            yield None

        _hop_size_bytes = (
            self._hop_size * self._audio_source.sw * self._audio_source.ch
        )
        cache = block[_hop_size_bytes:]
        yield block

        while True:
            block = self._audio_source.read(self._hop_size)
            if block:
                block = cache + block
                cache = block[_hop_size_bytes:]
                yield block
                continue
            yield None

    def read(self):
        try:
            block = next(self._blocks)
            if block == AudioIOError:
                raise AudioIOError("Audio Stream is not open.")
            return block
        except StopIteration:
            return None

    def rewind(self):
        super().rewind()
        self._blocks = self._iter_blocks_with_overlap()

    @property
    def hop_size(self):
        return self._hop_size

    @property
    def hop_dur(self):
        return self._hop_size / self.sr

    def __getattr__(self, name):
        return getattr(self._audio_source, name)


class AudioReader(DataSource):
    """
    A class to read fixed-size chunks of audio data from a source, which can
    be a file, standard input (with `input` set to "-"), or a microphone.
    Typically used by tokenization algorithms that require source objects with
    a `read` function to return data windows of consistent size, except for
    the last window if remaining data is insufficient.

    This class supports overlapping audio windows, recording the audio stream
    for later access (useful for microphone input), and limiting the maximum
    amount of data read.

    Parameters
    ----------
    input : str, bytes, AudioSource, AudioReader, AudioRegion, or None
        Input audio data. If a string, it should be the path to an audio file
        (use "-" for standard input). If bytes, the input is treated as raw
        audio data. If None, audio is read from a microphone. Any input that
        is not an :class:`AudioReader` will be converted, if possible, to an
        :class:`AudioSource` for processing. For raw audio (string path, bytes,
        or None), specify audio parameters using kwargs (`sampling_rate`,
        `sample_width`, `channels` or their aliases: `sr`, `sw`, `ch`).
    block_dur : float, default=0.01
        Duration of audio data (in seconds) to return in each `read` call.
    hop_dur : float, optional
        Duration of data to skip (in seconds) from the previous window. If set,
        it is used to calculate temporal overlap between the current and
        previous window (`overlap = block_dur - hop_dur`). If None (default),
        windows do not overlap.
    record : bool, default=False
        Whether to record audio data for later access. If True, recorded audio
        can be accessed using the `data` property after calling `rewind()`.
        Note: after `rewind()`, no new data is read from the source—subsequent
        `read` calls use the cached data.
    max_read : float, optional
        Maximum duration of audio data to read (in seconds). If None (default),
        data is read until the end of the stream or, for microphone input, until
        a Ctrl-C interruption.

    Additional audio parameters may be required if `input` is raw audio
    (None, bytes, or raw audio file):

    Other Parameters
    ----------------
    audio_format, fmt : str
        Type of audio data (e.g., wav, ogg, flac, raw). Used if `input` is a
        file path. If not provided, the format is inferred from the file
        extension or header.
    sampling_rate, sr : int
        Sampling rate of the audio data. Required for raw audio (bytes, None,
        or raw file).
    sample_width, sw : int
        Number of bytes per audio sample (typically 1, 2, or 4). Required for
        raw data.
    channels, ch : int
        Number of audio channels. Required for raw data.
    use_channel, uc : {None, "any", "mix", "avg", "average"} or int
        Specifies the channel used for split if `input` has multiple channels.
        All returned audio data includes data from *all* input channels. Options:

        - None or "any": Use any active channel, regardless of silence in others.
          (Default)
        - "mix" / "avg" / "average": Combine all channels by averaging.
        - int: Use the specified channel ID (0 <= value < `channels`).

    large_file : bool, default=False
        If True and `input` is a path to a *wav* or *raw* file, audio data is
        loaded lazily (one analysis window at a time). Otherwise, the entire
        file is loaded before processing. Use True for large files exceeding
        available memory.
    """

    def __init__(
        self,
        input,
        block_dur=0.01,
        hop_dur=None,
        record=False,
        max_read=None,
        **kwargs,
    ):
        if not isinstance(input, AudioSource):
            input = get_audio_source(input, **kwargs)
        self._record = record
        if record:
            input = _Recorder(input)
        if max_read is not None:
            input = _Limiter(input, max_read)
            self._max_read = max_read
        if hop_dur is None or hop_dur == block_dur:
            input = _FixedSizeAudioReader(input, block_dur)
        else:
            input = _OverlapAudioReader(input, block_dur, hop_dur)

        self._audio_source = input

    def __repr__(self):
        block_dur, hop_dur, max_read = None, None, None
        if self.block_dur is not None:
            block_dur = "{:.3f}".format(self.block_dur)
        if self.hop_dur is not None:
            hop_dur = "{:.3f}".format(self.hop_dur)
        if self.max_read is not None:
            max_read = "{:.3f}".format(self.max_read)
        return (
            "<{cls}(block_dur={block_dur}, "
            "hop_dur={hop_dur}, record={rewindable}, "
            "max_read={max_read})>"
        ).format(
            cls=self.__class__.__name__,
            block_dur=block_dur,
            hop_dur=hop_dur,
            rewindable=self._record,
            max_read=max_read,
        )

    @property
    def rewindable(self):
        return self._record

    @property
    def block_dur(self):
        return self._audio_source.block_size / self._audio_source.sr

    @property
    def hop_dur(self):
        if hasattr(self._audio_source, "hop_dur"):
            return self._audio_source.hop_size / self._audio_source.sr
        return self.block_dur

    @property
    def hop_size(self):
        if hasattr(self._audio_source, "hop_size"):
            return self._audio_source.hop_size
        return self.block_size

    @property
    def max_read(self):
        try:
            return self._audio_source.max_read
        except AttributeError:
            return None

    def read(self):
        return self._audio_source.read()

    def __getattr__(self, name):
        if name in ("data", "rewind") and not self.rewindable:
            raise AttributeError(
                "'AudioReader' has no attribute '{}'".format(name)
            )
        try:
            return getattr(self._audio_source, name)
        except AttributeError as exc:
            raise AttributeError(
                f"'AudioReader' has no attribute {name!r}"
            ) from exc


class Recorder(AudioReader):
    """
    A class to read fixed-size chunks of audio data from a source and store
    them in a cache. This class is equivalent to initializing
    :class:`AudioReader` with `record=True`. For more details on additional
    parameters, refer to :class:`AudioReader`.

    Once the desired amount of data is read, you can call the :meth:`rewind`
    method to access the recorded data via the :attr:`data` attribute. The
    cached data can also be re-read in fixed-size windows by calling
    :meth:`read`.
    """

    def __init__(
        self, input, block_dur=0.01, hop_dur=None, max_read=None, **kwargs
    ):
        super().__init__(
            input,
            block_dur=block_dur,
            hop_dur=hop_dur,
            record=True,
            max_read=max_read,
            **kwargs,
        )
