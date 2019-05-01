"""
This module gathers processing (i.e. tokenization) classes.

Class summary
=============

.. autosummary::

        StreamTokenizer
"""
import os
from auditok.util import AudioDataSource, DataValidator, AudioEnergyValidator
from auditok.io import check_audio_data, to_file

__all__ = ["split", "AudioRegion", "StreamTokenizer"]


DEFAULT_ANALYSIS_WINDOW = 0.05
DEFAULT_ENERGY_THRESHOLD = 50


def split(
    input,
    min_dur=0.2,
    max_dur=5,
    max_silence=0.3,
    drop_trailing_silence=False,
    strict_min_dur=False,
    analysis_window=0.01,
    **kwargs
):
    """Splits audio data and returns a generator of `AudioRegion`s
    TODO: implement max_trailing_silence

    :Parameters:

    input: str, bytes, AudioSource, AudioRegion, AudioDataSource
        input audio data. If str, it should be a path to an existing audio
        file. If bytes, input is considered as raw audio data.
    audio_format: str
        type of audio date (e.g., wav, ogg, raw, etc.). This will only be used
        if ´input´ is a string path to audio file. If not given, audio type
        will be guessed from file name extension or from file header.
    min_dur: float
        minimun duration in seconds of a detected audio event. Default: 0.2.
        Using large values, very short audio events (e.g., very short 1-word
        utterances like 'yes' or 'no') can be missed.
        Using very short values might result in a high number of short,
        unuseful audio events.
    max_dur: float
        maximum duration in seconds of a detected audio event. Default: 5.
    max_silence: float
        maximum duration of consecutive silence within an audio event. There
        might be many silent gaps of this duration within an audio event.
    drop_trailing_silence: bool
        drop trailing silence from detected events
    strict_min_dur: bool
        strict minimum duration. Drop an event if it is shorter than ´min_dur´
        even if it is continguous to the latest valid event. This happens if
        the the latest event had reached ´max_dur´.
    analysis_window: float
        duration of analysis window in seconds. Default: 0.05 second (50 ms).
        A value up to 0.1 second (100 ms) should be good for most use-cases.
        You might need a different value, especially if you use a custom
        validator.
    sampling_rate, sr: int
        sampling rate of audio data. Only needed for raw audio files/data.
    sample_width, sw: int
        number of bytes used to encode an audio sample, typically 1, 2 or 4.
        Only needed for raw audio files/data.
    channels, ch: int
        nuumber of channels of audio data. Only needed for raw audio files.
    use_channel, uc: int, str
        which channel to use if input has multichannel audio data. Can be an
        int (0 being the first channel), or one of the following special str
        values:
        - 'left': first channel (equivalent to 0)
        - 'right': second channel (equivalent to 1)
        - 'mix': compute average channel
        Default: 0, use the first channel.
    max_read: float
        maximum data to read in seconds. Default: `None`, read until there is
        no more data to read.
    validator: DataValidator
        custom data validator. If ´None´ (default), an `AudioEnergyValidor` is
        used with the given energy threshold.
    energy_threshold: float
        energy threshlod for audio activity detection, default: 50. If a custom
        validator is given, this argumemt will be ignored.
    """
    if isinstance(input, AudioDataSource):
        source = input
    else:
        block_dur = kwargs.get("analysis_window", DEFAULT_ANALYSIS_WINDOW)
        max_read = kwargs.get("max_read")
        params = kwargs.copy()
        if isinstance(input, AudioRegion):
            params["sampling_rate"] = input.sr
            params["sample_width"] = input.sw
            params["channels"] = input.ch
            input = bytes(input)

        source = AudioDataSource(
            input, block_dur=block_dur, max_read=max_read, **params
        )

    validator = kwargs.get("validator")
    if validator is None:
        energy_threshold = kwargs.get(
            "energy_threshold", kwargs.get("eth", DEFAULT_ENERGY_THRESHOLD)
        )
        validator = AudioEnergyValidator(source.sw, energy_threshold)

    mode = (
        StreamTokenizer.DROP_TRAILING_SILENCE if drop_trailing_silence else 0
    )
    if strict_min_dur:
        mode |= StreamTokenizer.STRICT_MIN_LENGTH

    min_length = _duration_to_nb_windows(min_dur, analysis_window)
    max_length = _duration_to_nb_windows(max_dur, analysis_window)
    max_continuous_silence = _duration_to_nb_windows(
        max_silence, analysis_window
    )

    print(min_length, max_length, max_continuous_silence)
    tokenizer = StreamTokenizer(
        validator, min_length, max_length, max_continuous_silence, mode=mode
    )
    source.open()
    token_gen = tokenizer.tokenize(source, generator=True)
    region_gen = (
        _make_audio_region(
            source.block_dur,
            token[1],
            token[0],
            source.sr,
            source.sw,
            source.ch,
        )
        for token in token_gen
    )
    return region_gen


def _duration_to_nb_windows(duration, analysis_window):
    """
    Converts a given duration into a positive integer on analysis windows.
    if `duration / analysis_window` is not an integer, the result will be
    rounded to the closest bigger integer. If `duration == 0`, returns `0`.
    `duration` and `analysis_window` can be in seconds or milliseconds but
    must be in the same unit.

    :Parameters:

    duration: float
        a given duration in seconds or ms
    analysis_window: float
        size of analysis window, in the same unit as `duration`

    Returns:
    --------
    nb_windows: int
        minimum number of `analysis_window`'s to cover `durartion`. That means
        that `analysis_window * nb_windows >= duration`.
    """
    if duration == 0:
        return 0
    if duration > analysis_window:
        nb_windows, rest = divmod(duration, analysis_window)
        if rest > 0:
            nb_windows += 1
        return int(nb_windows)


def _make_audio_region(
    frame_duration,
    start_frame,
    data_frames,
    sampling_rate,
    sample_width,
    channels,
):
    """Create and return an `AudioRegion`.

    :Parameters:

    frame_duration: float
        duration of analysis window in seconds
    start_frame: int
        index of the fisrt analysis window
    samling_rate: int
        sampling rate of audio data
    sample_width: int
        number of bytes of one audio sample
    channels: int
        number of channels of audio data

    Returns:
    audio_region: AudioRegion
        AudioRegion whose start time is calculeted as:
        `1000 * start_frame * frame_duration`
    """
    start = start_frame * frame_duration
    data = b"".join(data_frames)
    return AudioRegion(data, start, sampling_rate, sample_width, channels)


class AudioRegion(object):
    def __init__(self, data, start, sampling_rate, sample_width, channels):
        """
        A class for detected audio events.

        :Parameters:

            data: bytes
                audio data
            start: float
                start time in seconds
            samling_rate: int
                sampling rate of audio data
            sample_width: int
                number of bytes of one audio sample
            channels: int
                number of channels of audio data
        """
        check_audio_data(data, sample_width, channels)
        self._data = data
        self._start = start
        self._sampling_rate = sampling_rate
        self._sample_width = sample_width
        self._channels = channels

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self.start + self.duration

    @property
    def duration(self):
        """
        Returns region duration in seconds.
        """
        return len(self._data) / (
            self.sampling_rate * self.sample_width * self.channels
        )

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def sr(self):
        return self._sampling_rate

    @property
    def sample_width(self):
        return self._sample_width

    @property
    def sw(self):
        return self._sample_width

    @property
    def channels(self):
        return self._channels

    @property
    def ch(self):
        return self._channels

    def save(self, file, format=None, exists_ok=True, **audio_parameters):
        """Save audio region to file.

        :Parameters:

        file: str, file-like object
            path to output file or a file-like object. If ´str´, it may contain
            ´{start}´, ´{end}´ and ´{duration}´ place holders, they'll be
            replaced by region's ´start´, ´end´ and ´duration´ respectively.
            Example:

            .. code:: python
                region = AudioRegion(b'\0' * 2 * 24000,
                                     start=2.25,
                                     sampling_rate=16000,
                                     sample_width=2,
                                     channels=1)
                region.duration
                1.5
                region.end
                3.75
                region.save('audio_{start}-{end}.wav')
                audio_2.25-3.75.wav
                region.save('audio_{duration:.3f}_{start:.3f}-{end:.3f}.wav')
                audio_1.500_2.250-3.750.wav

        format: str
            type of audio file. If None (default), file type is guessed from
            `file`'s extension. If `file` is not a ´str´ or does not have
            an extension, audio data is as a raw (headerless) audio file.
        exists_ok: bool, default: True
            If True, overwrite ´file´ if a file with the same name exists.
            If False, raise an ´IOError´ if the file exists.
        audio_parameters: dict
            any keyword arguments to be passed to audio saving backend
            (e.g. bitrate, etc.)

        :Returns:

        file: str, file-like object
            name of the file of file-like object to which audio data was
            written. If parameter ´file´ was a ´str´ with at least one {start},
            {end} or {duration} place holders.

        :Raises:

        IOError if ´file´ exists and ´exists_ok´ is False.
        """
        if isinstance(file, str):
            file = file.format(
                start=round(self.start, 6),
                end=round(self.end, 6),
                duration=round(self.duration, 6)
            )
            if not exists_ok and os.path.exists(file):
                raise FileExistsError("file '{file}' exists".format(file=file))
        to_file(
            self._data,
            file,
            format,
            sr=self.sr,
            sw=self.sw,
            ch=self.ch,
            audio_parameters=audio_parameters
        )
        return file

    def __len__(self):
        """
        Rerurns region duration in milliseconds.
        """
        return round(self.duration * 1000)

    def __bytes__(self):
        return self._data

    def __repr__(self):
        return (
            "AudioRegion(data, start={:.3f}, end={:.3f}, "
            "sampling_rate={}, sample_width={}, channels={})".format(
                self.start, self.end, self.sr, self.sw, self.ch
            )
        )

    def __str__(self):
        return "AudioRegion(start={:.3f}, end={:.3f}, duration={:.3f}".format(
            self.start, self.end, self.duration
        )

    def __add__(self, other):
        """
        Concatenates this region and `other` and return a new region.
        Both regions must have the same sampling rate, sample width
        and number of channels. If not, raises a `ValueError`.
        """
        if not isinstance(other, AudioRegion):
            raise TypeError(
                "Can only concatenate AudioRegion, "
                'not "{}"'.format(type(other))
            )
        if other.sr != self.sr:
            raise ValueError(
                "Can only concatenate AudioRegions of the same "
                "sampling rate ({} != {})".format(self.sr, other.sr)
            )
        if other.sw != self.sw:
            raise ValueError(
                "Can only concatenate AudioRegions of the same "
                "sample width ({} != {})".format(self.sw, other.sw)
            )
        if other.ch != self.ch:
            raise ValueError(
                "Can only concatenate AudioRegions of the same "
                "number of channels ({} != {})".format(self.ch, other.ch)
            )
        data = self._data + other._data
        return AudioRegion(data, self.start, self.sr, self.sw, self.ch)

    def __radd__(self, other):
        """
        Concatenates `other` and this region. `other` should be an
        `AudioRegion` with the same audio parameters as this region
        but can exceptionally be `0` to make it possible to concatenate
        many regions with `sum`.
        """
        if other == 0:
            return self
        return other.add(self)

    def __getitem__(self, index):
        err_message = "AudioRegion index must a slice object without a step"
        if not isinstance(index, slice):
            raise TypeError(err_message)
        if index.step is not None:
            raise ValueError(err_message)

        start_ms = index.start if index.start is not None else 0
        stop_ms = index.stop if index.stop is not None else len(self)
        if not (isinstance(start_ms, int) and isinstance(stop_ms, int)):
            raise TypeError("Slicing Audioregion requires integers")

        if start_ms < 0:
            start_ms = max(start_ms + len(self), 0)
        if stop_ms < 0:
            stop_ms = max(stop_ms + len(self), 0)

        samples_per_ms = self.sr / 1000
        bytes_per_ms = samples_per_ms * self.sw * self.channels
        # if a fraction of a sample is covered, return the whole sample
        onset = int(start_ms * bytes_per_ms)
        offset = round(stop_ms * bytes_per_ms + 0.5)
        # recompute start_ms based on actual onset
        actual_start_s = onset / bytes_per_ms / 1000
        new_start = (
            self.start + actual_start_s
        )
        data = self._data[onset:offset]
        return AudioRegion(data, new_start, self.sr, self.sw, self.ch)


class StreamTokenizer:
    """
    Class for stream tokenizers. It implements a 4-state automaton scheme
    to extract sub-sequences of interest on the fly.

    :Parameters:

        `validator` :
            instance of `DataValidator` that implements `is_valid` method.

        `min_length` : *(int)*
            Minimum number of frames of a valid token. This includes all \
            tolerated non valid frames within the token.

        `max_length` : *(int)*
            Maximum number of frames of a valid token. This includes all \
            tolerated non valid frames within the token.

        `max_continuous_silence` : *(int)*
            Maximum number of consecutive non-valid frames within a token.
            Note that, within a valid token, there may be many tolerated \
            *silent* regions that contain each a number of non valid frames up to \
            `max_continuous_silence`

        `init_min` : *(int, default=0)*
            Minimum number of consecutive valid frames that must be **initially** \
            gathered before any sequence of non valid frames can be tolerated. This
            option is not always needed, it can be used to drop non-valid tokens as
            early as possible. **Default = 0** means that the option is by default 
            ineffective. 

        `init_max_silence` : *(int, default=0)*
            Maximum number of tolerated consecutive non-valid frames if the \
            number already gathered valid frames has not yet reached 'init_min'.
            This argument is normally used if `init_min` is used. **Default = 0**,
            by default this argument is not taken into consideration.

        `mode` : *(int, default=0)*
            `mode` can be:

        1. `StreamTokenizer.STRICT_MIN_LENGTH`: 
        if token *i* is delivered because `max_length`
        is reached, and token *i+1* is immediately adjacent to
        token *i* (i.e. token *i* ends at frame *k* and token *i+1* starts
        at frame *k+1*) then accept token *i+1* only of it has a size of at
        least `min_length`. The default behavior is to accept token *i+1*
        event if it is shorter than `min_length` (given that the above conditions
        are fulfilled of course).

        :Examples:

        In the following code, without `STRICT_MIN_LENGTH`, the 'BB' token is
        accepted although it is shorter than `min_length` (3), because it immediately
        follows the latest delivered token:

        .. code:: python

            from auditok import StreamTokenizer, StringDataSource, DataValidator

            class UpperCaseChecker(DataValidator):
                def is_valid(self, frame):
                    return frame.isupper()


            dsource = StringDataSource("aaaAAAABBbbb")
            tokenizer = StreamTokenizer(validator=UpperCaseChecker(),
                                        min_length=3,
                                        max_length=4,
                                        max_continuous_silence=0)

            tokenizer.tokenize(dsource)

        :output:

         .. code:: python

            [(['A', 'A', 'A', 'A'], 3, 6), (['B', 'B'], 7, 8)]


        The following tokenizer will however reject the 'BB' token:

        .. code:: python

            dsource = StringDataSource("aaaAAAABBbbb")
            tokenizer = StreamTokenizer(validator=UpperCaseChecker(), 
                                        min_length=3, max_length=4,
                                        max_continuous_silence=0,
                                        mode=StreamTokenizer.STRICT_MIN_LENGTH)
            tokenizer.tokenize(dsource)

        :output:

        .. code:: python

            [(['A', 'A', 'A', 'A'], 3, 6)]


        2. `StreamTokenizer.DROP_TRAILING_SILENCE`: drop all tailing non-valid frames
        from a token to be delivered if and only if it is not **truncated**.
        This can be a bit tricky. A token is actually delivered if:

        - a. `max_continuous_silence` is reached

        :or:

        - b. Its length reaches `max_length`. This is called a **truncated** token

        In the current implementation, a `StreamTokenizer`'s decision is only based on already seen
        data and on incoming data. Thus, if a token is truncated at a non-valid but tolerated
        frame (`max_length` is reached but `max_continuous_silence` not yet) any tailing
        silence will be kept because it can potentially be part of valid token (if `max_length`
        was bigger). But if `max_continuous_silence` is reached before `max_length`, the delivered
        token will not be considered as truncated but a result of *normal* end of detection
        (i.e. no more valid data). In that case the tailing silence can be removed if you use
        the `StreamTokenizer.DROP_TRAILING_SILENCE` mode.

        :Example:

        .. code:: python

             tokenizer = StreamTokenizer(validator=UpperCaseChecker(), min_length=3,
                                         max_length=6, max_continuous_silence=3,
                                         mode=StreamTokenizer.DROP_TRAILING_SILENCE)

             dsource = StringDataSource("aaaAAAaaaBBbbbb")
             tokenizer.tokenize(dsource)

        :output:

        .. code:: python

            [(['A', 'A', 'A', 'a', 'a', 'a'], 3, 8), (['B', 'B'], 9, 10)]

        The first token is delivered with its tailing silence because it is truncated
        while the second one has its tailing frames removed.

        Without `StreamTokenizer.DROP_TRAILING_SILENCE` the output would be:

        .. code:: python

            [(['A', 'A', 'A', 'a', 'a', 'a'], 3, 8), (['B', 'B', 'b', 'b', 'b'], 9, 13)]


        3. `StreamTokenizer.STRICT_MIN_LENGTH | StreamTokenizer.DROP_TRAILING_SILENCE`:
        use both options. That means: first remove tailing silence, then ckeck if the
        token still has at least a length of `min_length`.
    """

    SILENCE = 0
    POSSIBLE_SILENCE = 1
    POSSIBLE_NOISE = 2
    NOISE = 3

    STRICT_MIN_LENGTH = 2
    DROP_TRAILING_SILENCE = 4
    # alias
    DROP_TAILING_SILENCE = 4

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

        if not isinstance(validator, DataValidator):
            raise TypeError(
                "'validator' must be an instance of 'DataValidator'"
            )

        if max_length <= 0:
            raise ValueError(
                "'max_length' must be > 0 (value={0})".format(max_length)
            )

        if min_length <= 0 or min_length > max_length:
            raise ValueError(
                "'min_length' must be > 0 and <= 'max_length' (value={0})".format(
                    min_length
                )
            )

        if max_continuous_silence >= max_length:
            raise ValueError(
                "'max_continuous_silence' must be < 'max_length' (value={0})".format(
                    max_continuous_silence
                )
            )

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

        self._mode = None
        self.set_mode(mode)
        self._strict_min_length = (mode & self.STRICT_MIN_LENGTH) != 0
        self._drop_tailing_silence = (mode & self.DROP_TRAILING_SILENCE) != 0

        self._deliver = None
        self._tokens = None
        self._state = None
        self._data = None
        self._contiguous_token = False

        self._init_count = 0
        self._silence_length = 0
        self._start_frame = 0
        self._current_frame = 0

    def set_mode(self, mode):
        # TODO: use properties and make these deprecated
        """
        :Parameters:

            `mode` : *(int)*
                New mode, must be one of:


            - `StreamTokenizer.STRICT_MIN_LENGTH`

            - `StreamTokenizer.DROP_TRAILING_SILENCE`

            - `StreamTokenizer.STRICT_MIN_LENGTH | StreamTokenizer.DROP_TRAILING_SILENCE`

            - `0` TODO: this mode should have a name

        See `StreamTokenizer.__init__` for more information about the mode.
        """

        if not mode in [
            self.STRICT_MIN_LENGTH,
            self.DROP_TRAILING_SILENCE,
            self.STRICT_MIN_LENGTH | self.DROP_TRAILING_SILENCE,
            0,
        ]:

            raise ValueError("Wrong value for mode")

        self._mode = mode
        self._strict_min_length = (mode & self.STRICT_MIN_LENGTH) != 0
        self._drop_tailing_silence = (mode & self.DROP_TRAILING_SILENCE) != 0

    def get_mode(self):
        """
        Return the current mode. To check whether a specific mode is activated use
        the bitwise 'and' operator `&`. Example:

        .. code:: python 

            if mode & self.STRICT_MIN_LENGTH != 0:
               do_something()
        """
        return self._mode

    def _reinitialize(self):
        self._contiguous_token = False
        self._data = []
        self._tokens = []
        self._state = self.SILENCE
        self._current_frame = -1
        self._deliver = self._append_token

    def tokenize(self, data_source, callback=None, generator=False):
        """
        Read data from `data_source`, one frame a time, and process the read frames in
        order to detect sequences of frames that make up valid tokens.

        :Parameters:
           `data_source` : instance of the :class:`DataSource` class that implements a `read` method.
               'read' should return a slice of signal, i.e. frame (of whatever \
               type as long as it can be processed by validator) and None if \
               there is no more signal.

           `callback` : an optional 3-argument function.
               If a `callback` function is given, it will be called each time a valid token
               is found.


        :Returns:
           A list of tokens if `callback` is None. Each token is tuple with the following elements:

            .. code python

                (data, start, end)

           where `data` is a list of read frames, `start`: index of the first frame in the
           original data and `end` : index of the last frame. 

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

    def _process(self, frame):

        frame_is_valid = self.validator.is_valid(frame)

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
                # max token reached at this frame will _deliver if _contiguous_token
                # and not _strict_min_length
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
            and self._drop_tailing_silence
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
