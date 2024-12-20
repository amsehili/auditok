"""
Module for low-level audio input-output operations.

.. autosummary::
    :toctree: generated/

    AudioSource
    Rewindable
    BufferAudioSource
    WaveAudioSource
    PyAudioSource
    StdinAudioSource
    PyAudioPlayer
    from_file
    to_file
    player_for
"""

import os
import sys
import wave
from abc import ABC, abstractmethod

from .exceptions import AudioIOError, AudioParameterError

try:
    from pydub import AudioSegment

    _WITH_PYDUB = True
except ImportError:
    _WITH_PYDUB = False

try:
    from tqdm import tqdm as _tqdm

    DEFAULT_BAR_FORMAT_TQDM = "|" + "{bar}" + "|" + "[{elapsed}/{duration}]"
    DEFAULT_NCOLS_TQDM = 30
    DEFAULT_NCOLS_TQDM = 30
    DEFAULT_MIN_INTERVAL_TQDM = 0.05
    _WITH_TQDM = True
except ImportError:
    _WITH_TQDM = False


__all__ = [
    "AudioSource",
    "Rewindable",
    "BufferAudioSource",
    "RawAudioSource",
    "WaveAudioSource",
    "PyAudioSource",
    "StdinAudioSource",
    "PyAudioPlayer",
    "from_file",
    "to_file",
    "player_for",
]

DEFAULT_SAMPLING_RATE = 16000
DEFAULT_SAMPLE_WIDTH = 2
DEFAULT_NB_CHANNELS = 1


def check_audio_data(data, sample_width, channels):
    sample_size_bytes = int(sample_width * channels)
    nb_samples = len(data) // sample_size_bytes
    if nb_samples * sample_size_bytes != len(data):
        raise AudioParameterError(
            "The length of audio data must be an integer "
            "multiple of `sample_width * channels`"
        )


def _guess_audio_format(filename, fmt):
    """Guess the audio format from a file extension or normalize a provided
    format.

    This helper function attempts to determine the audio format based on the
    file extension of `filename` or by normalizing the format specified by the
    user in `fmt`.

    Parameters
    ----------
    filename : str or Path
        The audio file name, including its extension.
    fmt : str
        The un-normalized format provided by the user.

    Returns
    -------
    str or None
        The guessed audio format as a string, or None if no format could be
        determined.
    """

    if fmt is None:
        extension = os.path.splitext(filename)[1][1:].lower()
        if extension:
            fmt = extension
        else:
            return None
    fmt = fmt.lower()
    if fmt == "wave":
        fmt = "wav"
    return fmt


def _get_audio_parameters(param_dict):
    """
    Retrieve audio parameters from a dictionary of parameters.

    Each audio parameter can have a long name or a short name. If both are
    present, the long name takes precedence. If neither is found, an
    `AudioParameterError` is raised.

    Expected parameters:
        - `sampling_rate`, `sr` : int, the sampling rate.
        - `sample_width`, `sw` : int, the sample size in bytes.
        - `channels`, `ch` : int, the number of audio channels.

    Parameters
    ----------
    param_dict : dict
        A dictionary containing audio parameters, with possible keys as
        defined above.

    Returns
    -------
    tuple
        A tuple containing audio parameters as
        (sampling_rate, sample_width, channels).

    Raises
    ------
    AudioParameterError
        If a required parameter is missing, is not an integer, or is not a
        positive value.
    """

    parameters = []
    for long_name, short_name in (
        ("sampling_rate", "sr"),
        ("sample_width", "sw"),
        ("channels", "ch"),
    ):
        param = param_dict.get(long_name, param_dict.get(short_name))
        if param is None or not isinstance(param, int) or param <= 0:
            err_message = f"{long_name!r} (or {short_name!r}) must be a "
            err_message += f"positive integer, passed value: {param}."
            raise AudioParameterError(err_message)
        parameters.append(param)
    sampling_rate, sample_width, channels = parameters
    return sampling_rate, sample_width, channels


class AudioSource(ABC):
    """
    Base class for audio source objects.

    This class provides a foundation for audio source objects. Subclasses are
    expected to implement methods to open and close an audio stream, as well as
    to read the desired number of audio samples.

    Parameters
    ----------
    sampling_rate : int
        The number of samples per second of audio data.
    sample_width : int
        The size, in bytes, of each audio sample. Accepted values are 1, 2, or 4.
    channels : int
        The number of audio channels.
    """

    def __init__(
        self,
        sampling_rate,
        sample_width,
        channels,
    ):

        if sample_width not in (1, 2, 4):
            raise AudioParameterError(
                "Sample width must be one of: 1, 2 or 4 (bytes)"
            )

        self._sampling_rate = sampling_rate
        self._sample_width = sample_width
        self._channels = channels

    @abstractmethod
    def is_open(self):
        """Return True if audio source is open, False otherwise."""

    @abstractmethod
    def open(self):
        """Open audio source."""

    @abstractmethod
    def close(self):
        """Close audio source."""

    @abstractmethod
    def read(self, size):
        """Read and return up to `size` audio samples.

        This abstract method reads audio data and returns it as a bytes object,
        containing at most `size` samples.

        Parameters
        ----------
        size : int
            The number of samples to read.

        Returns
        -------
        bytes
            A bytes object containing the audio data, with a length of
            `N * sample_width * channels`, where `N` is:

            - `size`, if `size` is less than or equal to the number of remaining
            samples
            - the number of remaining samples, if `size` exceeds the remaining
            samples
        """

    @property
    def sampling_rate(self):
        """Number of samples per second of audio stream."""
        return self._sampling_rate

    @property
    def sr(self):
        """Number of samples per second of audio stream (alias for
        `sampling_rate)`."""
        return self._sampling_rate

    @property
    def sample_width(self):
        """Number of bytes used to represent one audio sample."""
        return self._sample_width

    @property
    def sw(self):
        """Number of bytes used to represent one audio sample (alias for
        `sample_width`)."""
        return self._sample_width

    @property
    def channels(self):
        """Number of channels in audio stream."""
        return self._channels

    @property
    def ch(self):
        """Number of channels in audio stream (alias for `channels`)."""
        return self.channels

    def __str__(self):
        return f"{self.__class__.__name__}(sampling_rate={self.sr}, sampling_rate={self.sw}, channels={self.ch})"  # noqa: B950

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}("
            f"sampling_rate={self.sampling_rate!r}, "
            f"sample_width={self.sample_width!r}, "
            f"channels={self.channels!r})>"
        )


class Rewindable(AudioSource):
    """Base class for rewindable audio sources.

    This class serves as a base for audio sources that support rewinding.
    Subclasses should implement a method to return to the beginning of the
    stream (`rewind`), and provide a property `position` that allows getting
    and setting the current stream position, expressed in number of samples.
    """

    @abstractmethod
    def rewind(self):
        """Go back to the beginning of audio stream."""

    @property
    @abstractmethod
    def position(self):
        """Return stream position in number of samples."""

    @position.setter
    @abstractmethod
    def position(self, position):
        """Set stream position in number of samples."""

    @property
    def position_s(self):
        """Return stream position in seconds."""
        return self.position / self.sampling_rate

    @position_s.setter
    def position_s(self, position_s):
        """Set stream position in seconds."""
        self.position = int(self.sampling_rate * position_s)

    @property
    def position_ms(self):
        """Return stream position in milliseconds."""
        return (self.position * 1000) // self.sampling_rate

    @position_ms.setter
    def position_ms(self, position_ms):
        """Set stream position in milliseconds."""
        if not isinstance(position_ms, int):
            raise ValueError("position_ms should be an int")
        self.position = int(self.sampling_rate * position_ms / 1000)


class BufferAudioSource(Rewindable):
    """An `AudioSource` that reads audio data from a memory buffer.

    This class implements the `Rewindable` interface, allowing audio data
    stored in a buffer to be read with support for rewinding and position
    control.

    Parameters
    ----------
    data : bytes
        The audio data stored in a memory buffer.
    sampling_rate : int, optional, default=16000
        The number of samples per second of audio data.
    sample_width : int, optional, default=2
        The size in bytes of one audio sample. Accepted values are 1, 2, or 4.
    channels : int, optional, default=1
        The number of audio channels.
    """

    def __init__(
        self,
        data,
        sampling_rate=16000,
        sample_width=2,
        channels=1,
    ):
        super().__init__(sampling_rate, sample_width, channels)
        check_audio_data(data, sample_width, channels)
        self._data = data
        self._sample_size_all_channels = sample_width * channels
        self._current_position_bytes = 0
        self._is_open = False

    def is_open(self):
        return self._is_open

    def open(self):
        self._is_open = True

    def close(self):
        self._is_open = False
        self.rewind()

    def read(self, size):
        if not self._is_open:
            raise AudioIOError("Stream is not open")
        if size is None or size < 0:
            offset = None
        else:
            bytes_to_read = self._sample_size_all_channels * size
            offset = self._current_position_bytes + bytes_to_read
        data = self._data[self._current_position_bytes : offset]
        if data:
            self._current_position_bytes += len(data)
            return data
        return None

    @property
    def data(self):
        """Get raw audio data as a `bytes` object."""
        return self._data

    def rewind(self):
        self.position = 0

    @property
    def position(self):
        """Get stream position in number of samples"""
        return self._current_position_bytes // self._sample_size_all_channels

    @position.setter
    def position(self, position):
        """Set stream position in number of samples."""
        position *= self._sample_size_all_channels
        if position < 0:
            position += len(self.data)
        if position < 0 or position > len(self.data):
            raise IndexError("Position out of range")
        self._current_position_bytes = position

    @property
    def position_ms(self):
        """Get stream position in milliseconds."""
        return (self._current_position_bytes * 1000) // (
            self._sample_size_all_channels * self.sampling_rate
        )

    @position_ms.setter
    def position_ms(self, position_ms):
        """Set stream position in milliseconds."""
        if not isinstance(position_ms, int):
            raise ValueError("position_ms should be an int")
        self.position = int(self.sampling_rate * position_ms / 1000)


class FileAudioSource(AudioSource):
    """Base class for `AudioSource`s that read audio data from a file.

    This class provides a foundation for audio sources that retrieve audio data
    from file sources.

    Parameters
    ----------
    sampling_rate : int, optional, default=16000
        The number of samples per second of audio data.
    sample_width : int, optional, default=2
        The size in bytes of one audio sample. Accepted values are 1, 2, or 4.
    channels : int, optional, default=1
        The number of audio channels.
    """

    def __init__(self, sampling_rate, sample_width, channels):
        super().__init__(sampling_rate, sample_width, channels)
        self._audio_stream = None

    def __del__(self):
        if self.is_open():
            self.close()

    def is_open(self):
        return self._audio_stream is not None

    def close(self):
        if self._audio_stream is not None:
            self._audio_stream.close()
            self._audio_stream = None

    @abstractmethod
    def _read_from_stream(self, size):
        """Read data from stream"""

    def read(self, size):
        if not self.is_open():
            raise AudioIOError("Audio stream is not open")
        data = self._read_from_stream(size)
        if not data:
            return None
        return data


class RawAudioSource(FileAudioSource):
    """
    An `AudioSource` class for reading data from a raw (headerless) audio file.

    This class is suitable for large raw audio files, allowing for efficient
    data handling without loading the entire file into memory.

    Parameters
    ----------
    filename : str or Path
        The path to the raw audio file.
    sampling_rate : int
        The number of samples per second of audio data.
    sample_width : int
        The size in bytes of each audio sample. Accepted values are 1, 2, or 4.
    channels : int
        The number of audio channels.
    """

    def __init__(self, filename, sampling_rate, sample_width, channels):
        super().__init__(sampling_rate, sample_width, channels)
        self._filename = filename
        self._audio_stream = None
        self._sample_size = sample_width * channels

    def open(self):
        if self._audio_stream is None:
            self._audio_stream = open(self._filename, "rb")

    def _read_from_stream(self, size):
        if size is None or size < 0:
            bytes_to_read = None
        else:
            bytes_to_read = size * self._sample_size
        data = self._audio_stream.read(bytes_to_read)
        return data


class WaveAudioSource(FileAudioSource):
    """
        An `AudioSource` class for reading data from a wave file.

    This class is suitable for large wave files, allowing for efficient data
    handling without loading the entire file into memory.

    Parameters
    ----------
    filename : str or Path
        The path to a valid wave file.
    """

    def __init__(self, filename):
        self._filename = str(filename)  # wave requires an str filename
        self._audio_stream = None
        stream = wave.open(self._filename, "rb")
        super().__init__(
            stream.getframerate(),
            stream.getsampwidth(),
            stream.getnchannels(),
        )
        stream.close()

    def open(self):
        if self._audio_stream is None:
            self._audio_stream = wave.open(self._filename)

    def _read_from_stream(self, size):
        if size is None or size < 0:
            size = -1
        return self._audio_stream.readframes(size)


class PyAudioSource(AudioSource):
    """An `AudioSource` class for reading data from a built-in microphone using
    PyAudio.

    This class leverages PyAudio (https://people.csail.mit.edu/hubert/pyaudio/)
    to capture audio data directly from a microphone.

    Parameters
    ----------
    sampling_rate : int, optional, default=16000
        The number of samples per second of audio data.
    sample_width : int, optional, default=2
        The size in bytes of each audio sample. Accepted values are 1, 2, or 4.
    channels : int, optional, default=1
        The number of audio channels.
    frames_per_buffer : int, optional, default=1024
        The number of frames per buffer, as specified by PyAudio.
    input_device_index : int or None, optional, default=None
        The PyAudio index of the audio device to read from. If None, the default
        audio device is used.
    """

    def __init__(
        self,
        sampling_rate=16000,
        sample_width=2,
        channels=1,
        frames_per_buffer=1024,
        input_device_index=None,
    ):

        super().__init__(sampling_rate, sample_width, channels)
        self._chunk_size = frames_per_buffer
        self.input_device_index = input_device_index

        import pyaudio

        self._pyaudio_object = pyaudio.PyAudio()
        self._pyaudio_format = self._pyaudio_object.get_format_from_width(
            self.sample_width
        )
        self._audio_stream = None

    def is_open(self):
        return self._audio_stream is not None

    def open(self):
        self._audio_stream = self._pyaudio_object.open(
            format=self._pyaudio_format,
            channels=self.channels,
            rate=self.sampling_rate,
            input=True,
            output=False,
            input_device_index=self.input_device_index,
            frames_per_buffer=self._chunk_size,
        )

    def close(self):
        if self._audio_stream is not None:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
            self._audio_stream = None

    def read(self, size):
        if self._audio_stream is None:
            raise IOError("Stream is not open")
        if self._audio_stream.is_active():
            data = self._audio_stream.read(size)
            if data is None or len(data) < 1:
                return None
            return data
        return None


class StdinAudioSource(FileAudioSource):
    """
    An `AudioSource` class for reading audio data from standard input.

    This class is designed to capture audio data directly from standard input,
    making it suitable for streaming audio sources.

    Parameters
    ----------
    sampling_rate : int, optional, default=16000
        The number of samples per second of audio data.
    sample_width : int, optional, default=2
        The size in bytes of each audio sample. Accepted values are 1, 2, or 4.
    channels : int, optional, default=1
        The number of audio channels.
    """

    def __init__(
        self,
        sampling_rate=16000,
        sample_width=2,
        channels=1,
    ):
        super().__init__(sampling_rate, sample_width, channels)
        self._is_open = False
        self._sample_size = sample_width * channels
        self._stream = sys.stdin.buffer

    def is_open(self):
        return self._is_open

    def open(self):
        self._is_open = True

    def close(self):
        self._is_open = False

    def _read_from_stream(self, size):
        bytes_to_read = size * self._sample_size
        data = self._stream.read(bytes_to_read)
        if data:
            return data
        return None


def _make_tqdm_progress_bar(iterable, total, duration, **tqdm_kwargs):
    fmt = tqdm_kwargs.get("bar_format", DEFAULT_BAR_FORMAT_TQDM)
    fmt = fmt.replace("{duration}", "{:.3f}".format(duration))
    tqdm_kwargs["bar_format"] = fmt

    tqdm_kwargs["ncols"] = tqdm_kwargs.get("ncols", DEFAULT_NCOLS_TQDM)
    tqdm_kwargs["mininterval"] = tqdm_kwargs.get(
        "mininterval", DEFAULT_MIN_INTERVAL_TQDM
    )
    return _tqdm(iterable, total=total, **tqdm_kwargs)


class PyAudioPlayer:
    """A class for audio playback using PyAudio.

    This class facilitates audio playback through the PyAudio library
    (https://people.csail.mit.edu/hubert/pyaudio/).

    Parameters
    ----------
    sampling_rate : int, optional, default=16000
        The number of samples per second of audio data.
    sample_width : int, optional, default=2
        The size in bytes of each audio sample. Accepted values are 1, 2, or 4.
    channels : int, optional, default=1
        The number of audio channels.
    """

    def __init__(
        self,
        sampling_rate=16000,
        sample_width=2,
        channels=1,
    ):
        if sample_width not in (1, 2, 4):
            raise ValueError("Sample width in bytes must be one of 1, 2 or 4")

        self.sampling_rate = sampling_rate
        self.sample_width = sample_width
        self.channels = channels

        import pyaudio

        self._p = pyaudio.PyAudio()
        self.stream = self._p.open(
            format=self._p.get_format_from_width(self.sample_width),
            channels=self.channels,
            rate=self.sampling_rate,
            input=False,
            output=True,
        )

    def play(self, data, progress_bar=False, **progress_bar_kwargs):
        chunk_gen, nb_chunks = self._chunk_data(data)
        if progress_bar and _WITH_TQDM:
            duration = len(data) / (
                self.sampling_rate * self.sample_width * self.channels
            )
            chunk_gen = _make_tqdm_progress_bar(
                chunk_gen,
                total=nb_chunks,
                duration=duration,
                **progress_bar_kwargs,
            )
        if self.stream.is_stopped():
            self.stream.start_stream()
        try:
            for chunk in chunk_gen:
                self.stream.write(chunk)
        except KeyboardInterrupt:
            pass
        self.stream.stop_stream()

    def stop(self):
        if not self.stream.is_stopped():
            self.stream.stop_stream()
        self.stream.close()
        self._p.terminate()

    def _chunk_data(self, data):
        # make audio chunks of 100 ms to allow interruption (like ctrl+c)
        bytes_1_sec = self.sampling_rate * self.sample_width * self.channels
        chunk_size = bytes_1_sec // 10
        # make sure chunk_size is a multiple of sample_width * channels
        chunk_size -= chunk_size % (self.sample_width * self.channels)
        nb_chunks, rest = divmod(len(data), chunk_size)
        if rest > 0:
            nb_chunks += 1
        chunk_gen = (
            data[i : i + chunk_size] for i in range(0, len(data), chunk_size)
        )
        return chunk_gen, nb_chunks


def player_for(source):
    """
    Return an `AudioPlayer` compatible with the specified `source`.

    This function creates an `AudioPlayer` instance (currently only
    `PyAudioPlayer` is implemented) that matches the audio properties of the
    provided `source`, ensuring compatibility in terms of sampling rate, sample
    width, and number of channels.

    Parameters
    ----------
    source : AudioSource
        An object with `sampling_rate`, `sample_width`, and `channels`
        attributes.

    Returns
    -------
    PyAudioPlayer
        An audio player with the same sampling rate, sample width, and number
        of channels as `source`.
    """

    return PyAudioPlayer(
        source.sampling_rate, source.sample_width, source.channels
    )


def get_audio_source(input=None, **kwargs):
    """
    Create and return an `AudioSource` based on the specified input.

    This function generates an `AudioSource` instance from various input types,
    allowing flexibility for audio data sources such as file paths, raw data,
    standard input, or microphone input via PyAudio.

    Parameters
    ----------
    input : str, bytes, "-", or None, optional
        The source to read audio data from. Possible values are:
        - `str`: Path to a valid audio file.
        - `bytes`: Raw audio data.
        - "-": Read raw data from standard input.
        - None (default): Read audio data from the microphone using PyAudio.
    kwargs : dict, optional
        Additional audio parameters used to construct the `AudioSource` object.
        Depending on the `input` type, these may be optional (e.g., for common
        audio file formats such as wav, ogg, or flac). When required, parameters
        include `sampling_rate`, `sample_width`, `channels`, or their short
        forms `sr`, `sw`, and `ch`. These parameters are typically needed when
        `input` is a path to a raw audio file, a bytes object with raw audio
        data, or None (for microphone input). See respective `AudioSource`
        classes for detailed parameter requirements.

    Returns
    -------
    AudioSource
        An audio source created based on the specified input and audio
        parameters.
    """

    if input == "-":
        return StdinAudioSource(*_get_audio_parameters(kwargs))

    if isinstance(input, bytes):
        return BufferAudioSource(input, *_get_audio_parameters(kwargs))

    # read data from a file
    if input is not None:
        return from_file(filename=input, **kwargs)

    # read data from microphone via pyaudio
    else:
        frames_per_buffer = kwargs.get("frames_per_buffer", 1024)
        input_device_index = kwargs.get("input_device_index")
        return PyAudioSource(
            *_get_audio_parameters(kwargs),
            frames_per_buffer=frames_per_buffer,
            input_device_index=input_device_index,
        )


def _load_raw(
    filename, sampling_rate, sample_width, channels, large_file=False
):
    """
    Load a raw audio file using standard Python file handling.

    This function loads audio data from a raw file. If `large_file` is set to
    True, it returns a `RawAudioSource` object that reads data lazily from disk.
    Otherwise, it loads all data into memory and returns a `BufferAudioSource`
    object.

    Parameters
    ----------
    filename : str or Path
        The path to the raw audio data file.
    sampling_rate : int
        The sampling rate of the audio data.
    sample_width : int
        The size, in bytes, of each audio sample.
    channels : int
        The number of audio channels.
    large_file : bool, optional
        If True, a `RawAudioSource` is returned to allow lazy data loading from
        disk. If False, returns a `BufferAudioSource` with all data loaded into
        memory.

    Returns
    -------
    AudioSource
        An `AudioSource` that reads data from the specified file. The source is
        either a `RawAudioSource` (for lazy loading) or a `BufferAudioSource`
        (for in-memory loading), depending on the value of `large_file`.
    """

    if None in (sampling_rate, sample_width, channels):
        raise AudioParameterError(
            "All audio parameters are required for raw audio files"
        )

    if large_file:
        return RawAudioSource(
            filename,
            sampling_rate=sampling_rate,
            sample_width=sample_width,
            channels=channels,
        )

    with open(filename, "rb") as fp:
        data = fp.read()
    return BufferAudioSource(
        data,
        sampling_rate=sampling_rate,
        sample_width=sample_width,
        channels=channels,
    )


def _load_wave(filename, large_file=False):
    """
    Load a wave audio file using standard Python module `wave`.

    This function loads audio data from a wave (.wav) file. If `large_file` is
    set to True, it returns a `WaveAudioSource` object that reads data lazily
    from disk. Otherwise, it loads all data into memory and returns a
    `BufferAudioSource` object.

    Parameters
    ----------
    filename : str or Path
        The path to the wave audio data file.
    large_file : bool, optional
        If True, a `WaveAudioSource` is returned to allow lazy data loading from
        disk. If False, returns a `BufferAudioSource` with all data loaded into
        memory.

    Returns
    -------
    AudioSource
        An `AudioSource` that reads data from the specified file. The source is
        either a `WaveAudioSource` (for lazy loading) or a `BufferAudioSource`
        (for in-memory loading), depending on the value of `large_file`.
    """

    if large_file:
        return WaveAudioSource(filename)
    with wave.open(str(filename)) as fp:
        channels = fp.getnchannels()
        srate = fp.getframerate()
        swidth = fp.getsampwidth()
        data = fp.readframes(-1)
    return BufferAudioSource(
        data, sampling_rate=srate, sample_width=swidth, channels=channels
    )


def _load_with_pydub(filename, audio_format=None):
    """
    Load audio from a compressed audio or video file using `pydub`.

    This function uses `pydub` to load compressed audio files. If a video file
    is specified, the audio track(s) are extracted and loaded.

    Parameters
    ----------
    filename : str or Path
        The path to the audio file.
    audio_format : str, optional, default=None
        The audio file format, if known (e.g., raw, webm, wav, ogg).

    Returns
    -------
    BufferAudioSource
        An `AudioSource` that reads data from the specified file.
    """

    func_dict = {
        "mp3": AudioSegment.from_mp3,
        "ogg": AudioSegment.from_ogg,
        "flv": AudioSegment.from_flv,
    }
    open_function = func_dict.get(audio_format, AudioSegment.from_file)
    segment = open_function(filename)
    return BufferAudioSource(
        data=segment.raw_data,
        sampling_rate=segment.frame_rate,
        sample_width=segment.sample_width,
        channels=segment.channels,
    )


def from_file(filename, audio_format=None, large_file=False, **kwargs):
    """Read audio data from `filename` and return an `AudioSource` object.

    If `audio_format` is None, the appropriate `AudioSource` class is inferred
    from the file extension. The `filename` can refer to a compressed audio or
    video file; if a video file is provided, its audio track(s) are extracted.
    This functionality requires `pydub` (https://github.com/jiaaro/pydub).

    By default, all audio data is loaded into memory to create a
    `BufferAudioSource` object, suitable for most cases. For very large files,
    set `large_file=True` to enable lazy loading, which reads audio data from
    disk each time `AudioSource.read` is called. Currently, lazy loading
    supports only wave and raw formats.

    If `audio_format` is `raw`, the following keyword arguments are required:

        - `sampling_rate`, `sr`: int, sampling rate of audio data.
        - `sample_width`, `sw`: int, size in bytes of one audio sample.
        - `channels`, `ch`: int, number of channels of audio data.

    See Also
    --------
    to_file : A related function for saving audio data to a file.

    Parameters
    ----------
    filename : str or Path
        The path to the input audio or video file.
    audio_format : str, optional
        The audio format (e.g., raw, webm, wav, ogg).
    large_file : bool, optional, default=False
        If True, the audio data is read lazily from disk rather than being
        fully loaded into memory.

    Other Parameters
    ----------------
    sampling_rate, sr : int
        The sampling rate of the audio data.
    sample_width : int
        The sample width in bytes (i.e., number of bytes per audio sample).
    channels : int
        The number of audio channels.

    Returns
    -------
    AudioSource
        An `AudioSource` object that reads data from the specified file.

    Raises
    ------
    AudioIOError
        If audio data cannot be read in the given format or if `audio_format`
        is `raw` and one or more required audio parameters are missing.
    """

    audio_format = _guess_audio_format(filename, audio_format)

    if audio_format == "raw":
        srate, swidth, channels = _get_audio_parameters(kwargs)
        return _load_raw(filename, srate, swidth, channels, large_file)

    if audio_format in ["wav", "wave"]:
        return _load_wave(filename, large_file)
    if large_file:
        err_msg = "if 'large_file` is True file format should be raw or wav"
        raise AudioIOError(err_msg)
    if _WITH_PYDUB:
        return _load_with_pydub(filename, audio_format=audio_format)
    else:
        raise AudioIOError(
            "pydub is required for audio formats other than raw or wav"
        )


def _save_raw(data, file):
    """
    Save audio data as a headerless (raw) file.

    This function writes audio data to a file in raw format, without any header
    information.

    Parameters
    ----------
    data : bytes
        The audio data to be saved.
    file : str or Path
        The path to the file where audio data will be saved.

    See Also
    --------
    to_file : A related function for saving audio data in various formats.
    """

    with open(file, "wb") as fp:
        fp.write(data)


def _save_wave(data, file, sampling_rate, sample_width, channels):
    """
    Save audio data to a wave file.

    This function writes audio data to a file in the wave format, including
    header information based on the specified audio parameters.

    Parameters
    ----------
    data : bytes
        The audio data to be saved.
    file : str or Path
        The path to the file where audio data will be saved.
    sampling_rate : int
        The sampling rate of the audio data.
    sample_width : int
        The size, in bytes, of each audio sample.
    channels : int
        The number of audio channels.

    See Also
    --------
    to_file : A related function for saving audio data in various formats.
    """

    if None in (sampling_rate, sample_width, channels):
        raise AudioParameterError(
            "All audio parameters are required to save wave audio files"
        )
    with wave.open(str(file), "w") as fp:
        fp.setframerate(sampling_rate)
        fp.setsampwidth(sample_width)
        fp.setnchannels(channels)
        fp.writeframes(data)


def _save_with_pydub(
    data, file, audio_format, sampling_rate, sample_width, channels
):
    """
    Save audio data using pydub.

    This function saves audio data to a file in various formats supported by
    pydub (https://github.com/jiaaro/pydub), such as mp3, wav, ogg, etc.

    Parameters
    ----------
    data : bytes
        The audio data to be saved.
    file : str or Path
        The path to the file where audio data will be saved.
    audio_format : str
        The audio format to save the file in (e.g., mp3, wav, ogg).
    sampling_rate : int
        The sampling rate of the audio data.
    sample_width : int
        The size, in bytes, of each audio sample.
    channels : int
        The number of audio channels.

    See Also
    --------
    to_file : A related function for saving audio data in various formats.
    """

    segment = AudioSegment(
        data,
        frame_rate=sampling_rate,
        sample_width=sample_width,
        channels=channels,
    )
    with open(file, "wb") as fp:
        segment.export(fp, format=audio_format)


def to_file(data, filename, audio_format=None, **kwargs):
    """
    Write audio data to a file.

    This function writes audio data to a file in the specified format. If
    `audio_format` is None, the output format will be inferred from the file
    extension. If `audio_format` is None and `filename` has no extension,
    the data will be saved as a raw audio file.

    Parameters
    ----------
    data : bytes-like
        The audio data to be written. Accepts `bytes`, `bytearray`, `memoryview`,
        `array`, or `numpy.ndarray` objects.
    filename : str or Path
        The path to the output audio file.
    audio_format : str, optional
        The audio format to use for saving the data (e.g., raw, webm, wav, ogg).
    kwargs : dict, optional
        Additional parameters required for non-raw audio formats:

        - `sampling_rate`, `sr` : int, the sampling rate of the audio data.
        - `sample_width`, `sw` : int, the size in bytes of one audio sample.
        - `channels`, `ch` : int, the number of audio channels.

    Raises
    ------
    AudioParameterError
        Raised if the output format is not raw and one or more required audio
        parameters are missing.
    AudioIOError
        Raised if the audio data cannot be written in the specified format.
    """

    audio_format = _guess_audio_format(filename, audio_format)
    if audio_format in (None, "raw"):
        _save_raw(data, filename)
        return
    sampling_rate, sample_width, channels = _get_audio_parameters(kwargs)
    if audio_format in ("wav", "wave"):
        _save_wave(data, filename, sampling_rate, sample_width, channels)
    elif _WITH_PYDUB:
        _save_with_pydub(
            data, filename, audio_format, sampling_rate, sample_width, channels
        )
    else:
        raise AudioIOError(
            f"cannot write file format {audio_format} (file name: {filename})"
        )
