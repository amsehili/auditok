"""
Module for low-level audio input-output operations.

Class summary
=============

.. autosummary::

        AudioSource
        Rewindable
        BufferAudioSource
        WaveAudioSource
        PyAudioSource
        StdinAudioSource
        PyAudioPlayer
        

Function summary
================

.. autosummary::

        from_file
        player_for
"""
import os
import sys
import wave
import warnings
import audioop
from array import array
from functools import partial

if sys.version_info >= (3, 0):
    PYTHON_3 = True
else:
    PYTHON_3 = False

try:
    from pydub import AudioSegment

    _WITH_PYDUB = True
except ImportError:
    _WITH_PYDUB = False

__all__ = [
    "AudioIOError",
    "AudioParameterError",
    "AudioSource",
    "Rewindable",
    "BufferAudioSource",
    "WaveAudioSource",
    "PyAudioSource",
    "StdinAudioSource",
    "PyAudioPlayer",
    "from_file",
    "player_for",
]

DEFAULT_SAMPLING_RATE = 16000
DEFAULT_SAMPLE_WIDTH = 2
DEFAULT_NB_CHANNELS = 1
DEFAULT_USE_CHANNEL = 0
DATA_FORMAT = {1: "b", 2: "h", 4: "i"}


class AudioIOError(Exception):
    pass


class AudioParameterError(AudioIOError):
    pass


def check_audio_data(data, sample_width, channels):
    sample_size_bytes = int(sample_width * channels)
    nb_samples = len(data) // sample_size_bytes
    if nb_samples * sample_size_bytes != len(data):
        raise AudioParameterError(
            "The length of audio data must be an integer "
            "multiple of `sample_width * channels`"
        )


def _guess_audio_format(fmt, filename):
    if fmt is None:
        extension = os.path.splitext(filename.lower())[1][1:]
        return extension if extension else None
    return fmt.lower()


def _normalize_use_channel(use_channel):
    """
    Returns a value of `use_channel` as expected by audio read/write fuctions.
    If `use_channel` is `None`, returns 0. If it's an integer, or the special
    str 'mix' returns it as is. If it's `left` or `right` returns 0 or 1
    respectively.
    """
    if use_channel is None:
        return 0
    if use_channel == "mix" or isinstance(use_channel, int):
        return use_channel
    try:
        return ["left", "right"].index(use_channel)
    except ValueError:
        err_message = "'use_channel' parameter must be an integer or one of "
        err_message += "('left', 'right', 'mix'), found: '{}'"
        raise AudioParameterError(err_message.format(use_channel))


def _get_audio_parameters(param_dict):
    """
    Gets audio parameters from a dictionary of parameters.
    A parameter can have a long name or a short name. If the long name is
    present, the short name is ignored. In neither is present then
    `AudioParameterError` is raised  except for the `use_channel` (or `uc`)
    parameter for which a defalut value of 0 is returned.

    Also raises `AudioParameterError` if sampling rate, sample width or
    channels is not an integer.

    Expected parameters are:

        `sampling_rate`, `sr`: int, sampling rate.
        `sample_width`, `sw`: int, sample size in bytes.
        `channels`, `ch`: int, number of channels.
        `use_channel`, `us`: int or str, which channel to use from data.
            Default value is 0 (first channel). The following special str
            values are also accepted:
                `left`: alias for 0
                `right`: alias for 1
                `mix`: indicates that all channels should be mixed up into one
                    single channel

    :Returns

        param_dict: tuple
            audio parameters as a tuple (sampling_rate,
                                         sample_width,
                                         channels,
                                         use_channel)
    """
    err_message = "'{ln}' (or '{sn}') must be a positive integer, found: '{val}'"
    parameters = []
    for (long_name, short_name) in (
        ("sampling_rate", "sr"),
        ("sample_width", "sw"),
        ("channels", "ch"),
    ):
        param = param_dict.get(long_name, param_dict.get(short_name))
        if param is None or not isinstance(param, int) or param <= 0:
            raise AudioParameterError(
                err_message.format(ln=long_name, sn=short_name, val=param)
            )
        parameters.append(param)
    sampling_rate, sample_width, channels = parameters
    use_channel = param_dict.get("use_channel", param_dict.get("uc", 0))
    use_channel = _normalize_use_channel(use_channel)
    return sampling_rate, sample_width, channels, use_channel


def _array_to_bytes(a):
    """
    Converts an `array.array` to `bytes`.
    """
    if PYTHON_3:
        return a.tobytes()
    else:
        return a.tostring()


def _mix_audio_channels(data, channels, sample_width):
    if channels == 1:
        return data
    if channels == 2:
        return audioop.tomono(data, sample_width, 0.5, 0.5)
    fmt = DATA_FORMAT[sample_width]
    buffer = array(fmt, data)
    mono_channels = [array(fmt, buffer[ch::channels]) for ch in range(channels)]
    avg_arr = array(fmt, (sum(samples) // channels for samples in zip(*mono_channels)))
    return _array_to_bytes(avg_arr)


def _extract_selected_channel(data, channels, sample_width, use_channel):
    if use_channel == "mix":
        return _mix_audio_channels(data, channels, sample_width)
    elif use_channel >= channels or use_channel < -channels:
        err_message = "use_channel == {} but audio data has only {} channel{}."
        err_message += " Selected channel must be 'mix' or an integer >= "
        err_message += "-channels and < channels"
        err_message = err_message.format(
            use_channel, channels, "s" if channels > 1 else ""
        )
        raise AudioParameterError(err_message)
    elif use_channel < 0:
        use_channel += channels
    fmt = DATA_FORMAT[sample_width]
    buffer = array(fmt, data)
    return _array_to_bytes(buffer[use_channel::channels])


class AudioSource:
    """ 
    Base class for audio source objects.

    Subclasses should implement methods to open/close and audio stream 
    and read the desired amount of audio samples.

    :Parameters:

        `sampling_rate` : int
            Number of samples per second of audio stream. Default = 16000.

        `sample_width` : int
            Size in bytes of one audio sample. Possible values : 1, 2, 4.
            Default = 2.

        `channels` : int
            Number of channels of audio stream.
    """

    def __init__(
        self,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        sample_width=DEFAULT_SAMPLE_WIDTH,
        channels=DEFAULT_NB_CHANNELS,
    ):

        if not sample_width in (1, 2, 4):
            raise AudioParameterError("Sample width must be one of: 1, 2 or 4 (bytes)")

        self._sampling_rate = sampling_rate
        self._sample_width = sample_width
        self._channels = channels

    def is_open(self):
        """ Return True if audio source is open, False otherwise """
        raise NotImplementedError

    def open(self):
        """ Open audio source """
        raise NotImplementedError

    def close(self):
        """ Close audio source """
        raise NotImplementedError

    def read(self, size):
        """
        Read and return `size` audio samples at most.

        :Parameters:

            `size` : int
                the number of samples to read.

        :Returns:

            Audio data as a string of length `N * sample_width * channels`,
            where `N` is:

            - `size` if `size` < 'left_samples'

            - 'left_samples' if `size` > 'left_samples' 
        """
        raise NotImplementedError

    def get_sampling_rate(self):
        """ Return the number of samples per second of audio stream """
        return self.sampling_rate

    @property
    def sampling_rate(self):
        """ Number of samples per second of audio stream """
        return self._sampling_rate

    @property
    def sr(self):
        """ Number of samples per second of audio stream """
        return self._sampling_rate

    def get_sample_width(self):
        """ Return the number of bytes used to represent one audio sample """
        return self.sample_width

    @property
    def sample_width(self):
        """ Number of bytes used to represent one audio sample """
        return self._sample_width

    @property
    def sw(self):
        """ Number of bytes used to represent one audio sample """
        return self._sample_width

    def get_channels(self):
        """ Return the number of channels of this audio source """
        return self.channels

    @property
    def channels(self):
        """ Number of channels of this audio source """
        return self._channels

    @property
    def ch(self):
        """ Return the number of channels of this audio source """
        return self.channels


class Rewindable(AudioSource):
    """
    Base class for rewindable audio streams.
    Subclasses should implement methods to return to the beginning of an
    audio stream as well as method to move to an absolute audio position
    expressed in time or in number of samples. 
    """

    @property
    def rewindable(self):
        return True

    def rewind(self):
        """ Go back to the beginning of audio stream """
        raise NotImplementedError

    @property
    def position(self):
        """Stream position in number of samples"""
        raise NotImplementedError

    @position.setter
    def position(self, position):
        raise NotImplementedError

    @property
    def position_s(self):
        """Stream position in seconds"""
        return self.position / self.sampling_rate

    @position_s.setter
    def position_s(self, position_s):
        self.position = int(self.sampling_rate * position_s)

    @property
    def position_ms(self):
        """Stream position in milliseconds"""
        return (self.position * 1000) // self.sampling_rate

    @position_ms.setter
    def position_ms(self, position_ms):
        if not isinstance(position_ms, int):
            raise ValueError("position_ms should be an int")
        self.position = int(self.sampling_rate * position_ms / 1000)

    def get_position(self):
        """ Return the total number of already read samples """
        warnings.warn(
            "'get_position' is deprecated, use 'position' property instead",
            DeprecationWarning,
        )
        return self.position

    def get_time_position(self):
        """ Return the total duration in seconds of already read data """
        warnings.warn(
            "'get_time_position' is deprecated, use 'position_s' or 'position_ms' properties instead",
            DeprecationWarning,
        )
        return self.position_s

    def set_position(self, position):
        """ Move to an absolute position 

        :Parameters:

            `position` : int
                number of samples to skip from the start of the stream
        """
        warnings.warn(
            "'set_position' is deprecated, set 'position' property instead",
            DeprecationWarning,
        )
        self.position = position

    def set_time_position(self, time_position):
        """ Move to an absolute position expressed in seconds

        :Parameters:

            `time_position` : float
                seconds to skip from the start of the stream
        """
        warnings.warn(
            "'set_time_position' is deprecated, set 'position_s' or 'position_ms' properties instead",
            DeprecationWarning,
        )
        self.position_s = time_position


class BufferAudioSource(Rewindable):
    """
    An :class:`AudioSource` that encapsulates and reads data from a memory buffer.
    It implements methods from :class:`Rewindable` and is therefore a navigable :class:`AudioSource`.
    """

    def __init__(
        self,
        data_buffer,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        sample_width=DEFAULT_SAMPLE_WIDTH,
        channels=DEFAULT_NB_CHANNELS,
    ):
        AudioSource.__init__(self, sampling_rate, sample_width, channels)
        check_audio_data(data_buffer, sample_width, channels)
        self._buffer = data_buffer
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
        bytes_to_read = self._sample_size_all_channels * size
        data = self._buffer[
            self._current_position_bytes : self._current_position_bytes + bytes_to_read
        ]
        if data:
            self._current_position_bytes += len(data)
            return data
        return None

    @property
    def data(self):
        return self._buffer

    def get_data_buffer(self):
        """ Return all audio data as one string buffer. """
        return self._buffer

    def set_data(self, data_buffer):
        """ Set new data for this audio stream. 

        :Parameters:

            `data_buffer` : str, basestring, Bytes
                a string buffer with a length multiple of (sample_width * channels)
        """
        check_audio_data(data_buffer, self.sample_width, self.channels)
        self._buffer = data_buffer
        self._current_position_bytes = 0

    def append_data(self, data_buffer):
        """ Append data to this audio stream

        :Parameters:

            `data_buffer` : str, basestring, Bytes
                a buffer with a length multiple of (sample_width * channels)
        """
        check_audio_data(data_buffer, self.sample_width, self.channels)
        self._buffer += data_buffer

    def rewind(self):
        self.set_position(0)

    @property
    def position(self):
        """Stream position in number of samples"""
        return self._current_position_bytes // self._sample_size_all_channels

    @position.setter
    def position(self, position):
        position *= self._sample_size_all_channels
        if position < 0:
            position += len(self.data)
        if position < 0 or position > len(self.data):
            raise IndexError("Position out of range")
        self._current_position_bytes = position

    @property
    def position_ms(self):
        """Stream position in milliseconds"""
        return (self._current_position_bytes * 1000) // (
            self._sample_size_all_channels * self.sampling_rate
        )

    @position_ms.setter
    def position_ms(self, position_ms):
        if not isinstance(position_ms, int):
            raise ValueError("position_ms should be an int")
        self.position = int(self.sampling_rate * position_ms / 1000)


class _FileAudioSource(AudioSource):
    def __init__(self, sampling_rate, sample_width, channels, use_channel):
        AudioSource.__init__(self, sampling_rate, sample_width, channels)
        self._audio_stream = None
        self._use_channel = _normalize_use_channel(use_channel)
        if channels > 1:
            self._extract_selected_channel = partial(
                _extract_selected_channel,
                channels=channels,
                sample_width=sample_width,
                use_channel=self._use_channel,
            )
        else:
            self._extract_selected_channel = lambda x: x

    def __del__(self):
        if self.is_open():
            self.close()

    @property
    def use_channel(self):
        return self._use_channel

    def is_open(self):
        return self._audio_stream is not None

    def close(self):
        if self._audio_stream is not None:
            self._audio_stream.close()
            self._audio_stream = None

    def _read_from_stream(self, size):
        raise NotImplementedError

    def read(self, size):
        if not self.is_open():
            raise AudioIOError("Audio stream is not open")
        data = self._read_from_stream(size)
        if data:
            return self._extract_selected_channel(data)
        return None


class RawAudioSource(_FileAudioSource, Rewindable):
    def __init__(self, file, sampling_rate, sample_width, channels, use_channel=0):
        _FileAudioSource.__init__(
            self, sampling_rate, sample_width, channels, use_channel
        )
        self._file = file
        self._audio_stream = None
        self._sample_size = sample_width * channels

    def open(self):
        if self._audio_stream is None:
            self._audio_stream = open(self._file, "rb")

    def _read_from_stream(self, size):
        bytes_to_read = size * self._sample_size
        data = self._audio_stream.read(bytes_to_read)
        return data


class WaveAudioSource(_FileAudioSource, Rewindable):
    """
    A class for an `AudioSource` that reads data from a wave file.
    This class should be used for large wave files to avoid loading
    the whole data to memory.

    :Parameters:

        `filename` :
            path to a valid wave file.
    """

    def __init__(self, filename, use_channel=0):
        self._filename = filename
        self._audio_stream = None
        stream = wave.open(self._filename, "rb")
        _FileAudioSource.__init__(
            self,
            stream.getframerate(),
            stream.getsampwidth(),
            stream.getnchannels(),
            use_channel,
        )
        stream.close()

    def open(self):
        if self._audio_stream is None:
            self._audio_stream = wave.open(self._filename)

    def _read_from_stream(self, size):
        return self._audio_stream.readframes(size)


class PyAudioSource(AudioSource):
    """
    A class for an `AudioSource` that reads data the built-in microphone using PyAudio. 
    """

    def __init__(
        self,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        sample_width=DEFAULT_SAMPLE_WIDTH,
        channels=DEFAULT_NB_CHANNELS,
        frames_per_buffer=1024,
        input_device_index=None,
    ):

        AudioSource.__init__(self, sampling_rate, sample_width, channels)
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


class StdinAudioSource(_FileAudioSource):
    """
    A class for an :class:`AudioSource` that reads data from standard input.
    """

    def __init__(
        self,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        sample_width=DEFAULT_SAMPLE_WIDTH,
        channels=DEFAULT_NB_CHANNELS,
        use_channel=0,
    ):

        _FileAudioSource.__init__(
            self, sampling_rate, sample_width, channels, use_channel
        )
        self._is_open = False
        self._sample_size = sample_width * channels
        if PYTHON_3:
            self._stream = sys.stdin.buffer
        else:
            self._stream = sys.stdin

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


class PyAudioPlayer:
    """
    A class for audio playback using Pyaudio
    """

    def __init__(
        self,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        sample_width=DEFAULT_SAMPLE_WIDTH,
        channels=DEFAULT_NB_CHANNELS,
    ):
        if not sample_width in (1, 2, 4):
            raise ValueError("Sample width must be one of: 1, 2 or 4 (bytes)")

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

    def play(self, data):
        if self.stream.is_stopped():
            self.stream.start_stream()

        for chunk in self._chunk_data(data):
            self.stream.write(chunk)

        self.stream.stop_stream()

    def stop(self):
        if not self.stream.is_stopped():
            self.stream.stop_stream()
        self.stream.close()
        self._p.terminate()

    def _chunk_data(self, data):
        # make audio chunks of 100 ms to allow interruption (like ctrl+c)
        chunk_size = int((self.sampling_rate * self.sample_width * self.channels) / 10)
        start = 0
        while start < len(data):
            yield data[start : start + chunk_size]
            start += chunk_size


def player_for(audio_source):
    """
    Return a :class:`PyAudioPlayer` that can play data from `audio_source`.

    :Parameters:

        `audio_source` : 
            an `AudioSource` object.

    :Returns:

        `PyAudioPlayer` that has the same sampling rate, sample width and number of channels
        as `audio_source`.
    """

    return PyAudioPlayer(
        audio_source.get_sampling_rate(),
        audio_source.get_sample_width(),
        audio_source.get_channels(),
    )


def get_audio_source(input=None, **kwargs):
    """
    Create and return an AudioSource from input.

    Parameters:

        ´input´ : str, bytes, "-" or None
        Source to read audio data from. If str, it should be a path to a valid
        audio file. If bytes, it is interpreted as raw audio data. if equals to
        "-", raw data will be read from stdin. If None, read audio data from
        microphone using PyAudio.
    """

    sampling_rate = kwargs.get("sampling_rate", kwargs.get("sr", DEFAULT_SAMPLING_RATE))
    sample_width = kwargs.get("sample_rate", kwargs.get("sw", DEFAULT_SAMPLE_WIDTH))
    channels = kwargs.get("channels", kwargs.get("ch", DEFAULT_NB_CHANNELS))
    use_channel = kwargs.get("use_channel", kwargs.get("uc", DEFAULT_USE_CHANNEL))
    if input == "-":
        return StdinAudioSource(sampling_rate, sample_width, channels, use_channel)

    if isinstance(input, bytes):
        return BufferAudioSource(input, sampling_rate, sample_width, channels)

    # read data from a file
    if input is not None:
        return from_file(filename=input, **kwargs)

    # read data from microphone via pyaudio
    else:
        frames_per_buffer = kwargs.get("frames_per_buffer", 1024)
        input_device_index = kwargs.get("input_device_index")
        return PyAudioSource(
            sampling_rate=sampling_rate,
            sample_width=sample_width,
            channels=channels,
            frames_per_buffer=frames_per_buffer,
            input_device_index=input_device_index,
        )


def _load_raw(
    file, sampling_rate, sample_width, channels, use_channel=0, large_file=False
):
    """
    Load a raw audio file with standard Python.
    If `large_file` is True, audio data will be lazily
    loaded to memory.

    See also :func:`from_file`.

    :Parameters:
        `file` : filelike object or str
            raw audio file to open
        `sampling_rate`: int
            sampling rate of audio data
        `sample_width`: int
            sample width of audio data
        `channels`: int
            number of channels of audio data
        `use_channel`: int
            audio channel to read if file is not mono audio. This must be an integer
            0 >= and < channels, or one of 'left' (treated as 0 or first channel), or
            right (treated as 1 or second channels). 

    :Returns:

        `PyAudioPlayer` that has the same sampling rate, sample width and number of channels
        as `audio_source`.
    """
    if None in (sampling_rate, sample_width, channels):
        raise AudioParameterError(
            "All audio parameters are required for raw audio files"
        )

    if large_file:
        return RawAudioSource(
            file,
            sampling_rate=sampling_rate,
            sample_width=sample_width,
            channels=channels,
            use_channel=use_channel,
        )
    else:
        with open(file, "rb") as fp:
            data = fp.read()
        if channels != 1:
            # TODO check if striding with mmap doesn't load all data to memory
            data = _extract_selected_channel(data, channels, sample_width, use_channel)
        return BufferAudioSource(
            data, sampling_rate=sampling_rate, sample_width=sample_width, channels=1
        )


def _load_wave(filename, large_file=False, use_channel=0):
    """
    Load a wave audio file with standard Python.
    If `large_file` is True, audio data will be lazily
    loaded to memory.

    See also :func:`to_file`.
    """
    if large_file:
        return WaveAudioSource(filename, use_channel)
    with wave.open(filename) as fp:
        channels = fp.getnchannels()
        srate = fp.getframerate()
        swidth = fp.getsampwidth()
        data = fp.readframes(-1)
    if channels > 1:
        data = _extract_selected_channel(data, channels, swidth, use_channel)
    return BufferAudioSource(data, sampling_rate=srate, sample_width=swidth, channels=1)


def _load_with_pydub(filename, audio_format, use_channel=0):
    """Open compressed audio file using pydub. If a video file
    is passed, its audio track(s) are extracted and loaded.
    This function should not be called directely, use :func:`from_file`
    instead.

    :Parameters:

    `filename`:
        path to audio file.
    `audio_format`:
        string, audio file format (e.g. raw, webm, wav, ogg)
    """
    func_dict = {
        "mp3": AudioSegment.from_mp3,
        "ogg": AudioSegment.from_ogg,
        "flv": AudioSegment.from_flv,
    }
    open_function = func_dict.get(audio_format, AudioSegment.from_file)
    segment = open_function(filename)
    data = segment._data
    if segment.channels > 1:
        data = _extract_selected_channel(
            data, segment.channels, segment.sample_width, use_channel
        )
    return BufferAudioSource(
        data_buffer=data,
        sampling_rate=segment.frame_rate,
        sample_width=segment.sample_width,
        channels=1,
    )


def from_file(filename, audio_format=None, large_file=False, **kwargs):
    """
    Read audio data from `filename` and return an `AudioSource` object.
    if `audio_format` is None, the appropriate :class:`AudioSource` class is
    guessed from file's extension. `filename` can be a compressed audio or
    video file. This will require installing pydub:
    (https://github.com/jiaaro/pydub).

    The normal behavior is to load all audio data to memory from which a
    :class:`BufferAudioSource` object is created. This should be convenient
    most     of the time unless audio file is very large. In that case, and
    in order to load audio data in lazy manner (i.e. read data from disk each
    time :func:`AudioSource.read` is called), `large_file` should be True.

    Note that the current implementation supports only wave and raw formats for
    lazy audio loading.

    See also :func:`to_file`.

    :Parameters:

    `filename`: str
        path to input audio or video file.
    `audio_format`: str
        audio format used to save data  (e.g. raw, webm, wav, ogg)
    `large_file`: bool
        If True, audio won't fully be loaded to memory but only when a window
        is read disk.

    :kwargs:

    If an audio format other than `raw` is used, the following keyword
    arguments are required:

    `sampling_rate`, `sr`: int
        sampling rate of audio data
    `sample_width`: int
        sample width (i.e. number of bytes used to represent one audio sample)
    `channels`: int
        number of channels of audio data
    `use_channel`: int, str
        audio channel to extract from input file if file is not mono audio.
        This must be an integer >= 0 and < channels, or one of the special
        values `left` and `right` (treated as 0 and 1 respectively).

    :Returns:

    An `AudioSource` object that reads data from input file.

    :Raises:

    An `AudioIOError` is raised if audio data cannot be read in the given
    format; or if format is `raw` and one or more audio parameters are missing.
    """
    audio_format = _guess_audio_format(audio_format, filename)

    if audio_format == "raw":
        srate, swidth, channels, use_channel = _get_audio_parameters(kwargs)
        return _load_raw(filename, srate, swidth, channels, use_channel, large_file)

    use_channel = _normalize_use_channel(kwargs.get("use_channel"))
    if audio_format in ["wav", "wave"]:
        return _load_wave(filename, large_file, use_channel)
    if large_file:
        raise AudioIOError("Large file format should be raw or wav")
    if _WITH_PYDUB:
        return _load_with_pydub(
            filename, audio_format=audio_format, use_channel=use_channel
        )
    else:
        raise AudioIOError("pydub is required for audio formats other than raw or wav")


def _save_raw(data, file):
    """
    Saves audio data as a headerless (i.e. raw) file.
    See also :func:`to_file`.
    """
    with open(file, "wb") as fp:
        fp.write(data)


def _save_wave(data, file, sampling_rate, sample_width, channels):
    """
    Saves audio data to a wave file.
    See also :func:`to_file`.
    """
    if None in (sampling_rate, sample_width, channels):
        raise AudioParameterError(
            "All audio parameters are required to save wave audio files"
        )
    with wave.open(file, "w") as fp:
        fp.setframerate(sampling_rate)
        fp.setsampwidth(sample_width)
        fp.setnchannels(channels)
        fp.writeframes(data)


def _save_with_pydub(data, file, audio_format, sampling_rate, sample_width, channels):
    """
    Saves audio data with pydub (https://github.com/jiaaro/pydub).
    See also :func:`to_file`.
    """
    segment = AudioSegment(
        data, frame_rate=sampling_rate, sample_width=sample_width, channels=channels
    )
    with open(file, "wb") as fp:
        segment.export(fp, format=audio_format)


def to_file(data, file, audio_format=None, **kwargs):
    """
    Writes audio data to file. If `audio_format` is `None`, output
    audio format will be guessed from extension. If `audio_format`
    is `None` and `file` comes without an extension then audio
    data will be written as a raw audio file.

    :Parameters:

        `data`: buffer of bytes
            audio data to be written. Can be a `bytes`, `bytearray`,
            `memoryview`, `array` or `numpy.ndarray` object.
        `file`: str
            path to output audio file
        `audio_format`: str
            audio format used to save data (e.g. raw, webm, wav, ogg)
        :kwargs:
            If an audio format other than raw is used, the following
            keyword arguments are required:
            `sampling_rate`, `sr`: int
                sampling rate of audio data
            `sample_width`, `sw`: int
                sample width (i.e., number of bytes of one audio sample)
            `channels`, `ch`: int
                number of channels of audio data
    :Raises:

        `AudioParameterError` if output format is different than raw and one
        or more audio parameters are missing.
        `AudioIOError` if audio data cannot be written in the desired format.
    """
    audio_format = _guess_audio_format(audio_format, file)
    if audio_format in (None, "raw"):
        _save_raw(data, file)
        return
    try:
        params = _get_audio_parameters(kwargs)
        sampling_rate, sample_width, channels, _ = params
    except AudioParameterError as exc:
        err_message = "All audio parameters are required to save formats "
        "other than raw. Error detail: {}".format(exc)
        raise AudioParameterError(err_message)
    if audio_format in ("wav", "wave"):
        _save_wave(data, file, sampling_rate, sample_width, channels)
    elif _WITH_PYDUB:
        _save_with_pydub(
            data, file, audio_format, sampling_rate, sample_width, channels
        )
    else:
        err_message = "cannot write file format {} (file name: {})"
        raise AudioIOError(err_message.format(audio_format, file))
