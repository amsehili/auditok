"""
This module gathers processing (i.e. tokenization) classes.

Class summary
=============

.. autosummary::

        StreamTokenizer
"""
import os
import math
from auditok.util import AudioDataSource, DataValidator, AudioEnergyValidator
from auditok.io import check_audio_data, to_file, player_for, get_audio_source
from auditok.exceptions import TooSamllBlockDuration

try:
    from auditok.plotting import plot, plot_detections

    _WITH_MATPLOTLIB = True
except ImportError:
    _WITH_MATPLOTLIB = False


try:
    from . import signal_numpy as signal
except ImportError:
    from . import signal

__all__ = ["split", "AudioRegion", "StreamTokenizer"]


DEFAULT_ANALYSIS_WINDOW = 0.05
DEFAULT_ENERGY_THRESHOLD = 50
_EPSILON = 1e-6


def split(
    input,
    min_dur=0.2,
    max_dur=5,
    max_silence=0.3,
    drop_trailing_silence=False,
    strict_min_dur=False,
    **kwargs
):
    """Splits audio data and returns a generator of `AudioRegion`s
    TODO: implement max_trailing_silence

    :Parameters:

    input: str, bytes, AudioSource, AudioRegion, AudioDataSource
        input audio data. If str, it should be a path to an existing audio
        file. If bytes, input is considered as raw audio data.
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
    analysis_window, aw: float
        duration of analysis window in seconds. Default: 0.05 second (50 ms).
        A value up to 0.1 second (100 ms) should be good for most use-cases.
        You might need a different value, especially if you use a custom
        validator.
    audio_format, fmt: str
        type of audio date (e.g., wav, ogg, raw, etc.). This will only be used
        if ´input´ is a string path to audio file. If not given, audio type
        will be guessed from file name extension or from file header.
    sampling_rate, sr: int
        sampling rate of audio data. Only needed for raw audio files/data.
    sample_width, sw: int
        number of bytes used to encode an audio sample, typically 1, 2 or 4.
        Only needed for raw audio files/data.
    channels, ch: int
        nuumber of channels of audio data. Only needed for raw audio files.
    use_channel, uc: int, str
        which channel to use if input has multichannel audio data. Can be an
        int (0 being the first channel), or one of the following values:
            - None, "any": a valid frame from one any given channel makes
              parallel frames from all other channels automatically valid.
            - 'mix': compute average channel (i.e. mix down all channels)
    max_read, mr: float
        maximum data to read in seconds. Default: `None`, read until there is
        no more data to read.
    validator, val: DataValidator
        custom data validator. If ´None´ (default), an `AudioEnergyValidor` is
        used with the given energy threshold.
    energy_threshold, eth: float
        energy threshlod for audio activity detection, default: 50. If a custom
        validator is given, this argumemt will be ignored.
    """
    if min_dur <= 0:
        raise ValueError("'min_dur' ({}) must be > 0".format(min_dur))
    if max_dur <= 0:
        raise ValueError("'max_dur' ({}) must be > 0".format(max_dur))
    if max_silence < 0:
        raise ValueError("'max_silence' ({}) must be >= 0".format(max_silence))

    if isinstance(input, AudioDataSource):
        source = input
        analysis_window = source.block_dur
    else:
        analysis_window = kwargs.get(
            "analysis_window", kwargs.get("aw", DEFAULT_ANALYSIS_WINDOW)
        )
        if analysis_window <= 0:
            raise ValueError(
                "'analysis_window' ({}) must be > 0".format(analysis_window)
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
            source = AudioDataSource(
                input, block_dur=analysis_window, **params
            )
        except TooSamllBlockDuration as exc:
            err_msg = "Too small 'analysis_windows' ({0}) for sampling rate "
            err_msg += "({1}). Analysis windows should at least be 1/{1} to "
            err_msg += "cover one single data sample"
            raise ValueError(err_msg.format(exc.block_dur, exc.sampling_rate))

    validator = kwargs.get("validator", kwargs.get("val"))
    if validator is None:
        energy_threshold = kwargs.get(
            "energy_threshold", kwargs.get("eth", DEFAULT_ENERGY_THRESHOLD)
        )
        use_channel = kwargs.get("use_channel", kwargs.get("uc"))
        validator = AudioEnergyValidator(
            energy_threshold, source.sw, source.ch, use_channel=use_channel
        )
    mode = (
        StreamTokenizer.DROP_TRAILING_SILENCE if drop_trailing_silence else 0
    )
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

    # print(min_length, max_length, max_continuous_silence)
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


def _duration_to_nb_windows(
    duration, analysis_window, round_fn=round, epsilon=0
):
    """
    Converts a given duration into a positive integer of analysis windows.
    if `duration / analysis_window` is not an integer, the result will be
    rounded to the closest bigger integer. If `duration == 0`, returns `0`.
    If `duration < analysis_window`, returns 1.
    `duration` and `analysis_window` can be in seconds or milliseconds but
    must be in the same unit.

    :Parameters:

    duration: float
        a given duration in seconds or ms.
    analysis_window: float
        size of analysis window, in the same unit as `duration`.
    round_fn: callable
        function called to round the result. Default: `round`.
    epsilon: float
        small value to add to the division result before rounding.
        E.g., `0.3 / 0.1 = 2.9999999999999996`, when called with
        `round_fn=math.floor` returns `2` instead of `3`. Adding a small value
        to `0.3 / 0.1` avoids this error.

    Returns:
    --------
    nb_windows: int
        minimum number of `analysis_window`'s to cover `durartion`. That means
        that `analysis_window * nb_windows >= duration`.
    """
    if duration < 0 or analysis_window <= 0:
        err_msg = "'duration' ({}) must be >= 0 and 'analysis_window' ({}) > 0"
        raise ValueError(err_msg.format(duration, analysis_window))
    if duration == 0:
        return 0
    return int(round_fn(duration / analysis_window + epsilon))


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
    duration = len(data) / (sampling_rate * sample_width * channels)
    meta = {"start": start, "end": start + duration}
    return AudioRegion(data, sampling_rate, sample_width, channels, meta)


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
    def __getitem__(self, index):
        err_msg = (
            "Slicing AudioRegion by milliseconds requires indices of type "
        )
        err_msg += "'int' without a step (e.g. region.sec[500:1500])"
        start_ms, stop_ms = _check_convert_index(index, (int), err_msg)
        start_sec = start_ms / 1000
        stop_sec = None if stop_ms is None else stop_ms / 1000
        index = slice(start_sec, stop_sec)
        return super(_MillisView, self).__getitem__(index)

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
    def __getattr__(self, name):
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


class AudioRegion(object):
    def __init__(self, data, sampling_rate, sample_width, channels, meta=None):
        """
        A class for detected audio events.

        :Parameters:

            data: bytes
                audio data
            samling_rate: int
                sampling rate of audio data
            sample_width: int
                number of bytes of one audio sample
            channels: int
                number of channels of audio data
        """
        check_audio_data(data, sample_width, channels)
        self._data = data
        self._sampling_rate = sampling_rate
        self._sample_width = sample_width
        self._channels = channels
        self._samples = None
        self.splitp = self.split_and_plot

        if meta is not None:
            self._meta = _AudioRegionMetadata(meta)
        else:
            self._meta = None

        self._seconds_view = _SecondsView(self)
        self.s = self.sec

        self._millis_view = _MillisView(self)
        self.ms = self.millis

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, new_meta):
        self._meta = _AudioRegionMetadata(new_meta)

    @classmethod
    def load(cls, file, skip=0, max_read=None, **kwargs):
        audio_source = get_audio_source(file, **kwargs)
        audio_source.open()
        if skip is not None and skip > 0:
            skip_samples = int(skip * audio_source.sampling_rate)
            audio_source.read(skip_samples)
        if max_read is None or max_read < 0:
            max_read_samples = None
        else:
            max_read_samples = round(max_read * audio_source.sampling_rate)
        data = audio_source.read(max_read_samples)
        audio_source.close()
        return cls(
            data,
            audio_source.sampling_rate,
            audio_source.sample_width,
            audio_source.channels,
        )

    @property
    def sec(self):
        return self._seconds_view

    @property
    def millis(self):
        return self._millis_view

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

    def play(self, progress_bar=False, player=None, **progress_bar_kwargs):
        """Play audio region

        :Parameters:

        player: AudioPalyer, default: None
            audio player to use. if None (default), use `player_for(self)`
            to get a new audio player.

        progress_bar bool, default: False
            whether to use a progress bar while playing audio. Default: False.

        progress_bar_kwargs: kwargs
            keyword arguments to pass to progress_bar object. Currently only
            `tqdm` is supported.
        """
        if player is None:
            player = player_for(self)
        player.play(
            self._data, progress_bar=progress_bar, **progress_bar_kwargs
        )

    def save(self, file, format=None, exists_ok=True, **audio_parameters):
        """Save audio region to file.

        :Parameters:

        file: str, file-like object
            path to output file or a file-like object. If ´str´, it may contain
            and ´{duration}´ place holders as well as any place holder that
            this region's metadata might contain (e.g., ´{meta.start}´).


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

        Example:

        .. code:: python
            region = AudioRegion(b'\0' * 2 * 24000,
                                    sampling_rate=16000,
                                    sample_width=2,
                                    channels=1)
            region.meta = {"start": 2.25, "end": 2.25 + region.duration}
            region.save('audio_{meta.start}-{meta.end}.wav')
            audio_2.25-3.75.wav
            region.save('region_{meta.start:.3f}_{duration:.3f}.wav')
            audio_2.250_1.500.wav
        """
        if isinstance(file, str):
            file = file.format(duration=self.duration, meta=self.meta)
            if not exists_ok and os.path.exists(file):
                raise FileExistsError("file '{file}' exists".format(file=file))
        to_file(
            self._data,
            file,
            format,
            sr=self.sr,
            sw=self.sw,
            ch=self.ch,
            audio_parameters=audio_parameters,
        )
        return file

    def split(
        self,
        min_dur=0.2,
        max_dur=5,
        max_silence=0.3,
        drop_trailing_silence=False,
        strict_min_dur=False,
        **kwargs
    ):
        """Split region. See :auditok.split() for split parameters description.
        """
        return split(
            self,
            min_dur=min_dur,
            max_dur=max_dur,
            max_silence=max_silence,
            drop_trailing_silence=drop_trailing_silence,
            strict_min_dur=strict_min_dur,
            **kwargs
        )

    def plot(self, show=True, **plot_kwargs):
        if _WITH_MATPLOTLIB:
            plot(self, self.sr, show=show, **plot_kwargs)
        else:
            raise RuntimeWarning("Plotting requires matplotlib")

    def split_and_plot(
        self,
        min_dur=0.2,
        max_dur=5,
        max_silence=0.3,
        drop_trailing_silence=False,
        strict_min_dur=False,
        show=True,
        plot_kwargs=None,
        **kwargs
    ):
        """Split region and plot signal and detection. Alias: `splitp`.
        See :auditok.split() for split parameters description.
        """
        if _WITH_MATPLOTLIB:
            regions = split(
                self,
                min_dur=min_dur,
                max_dur=max_dur,
                max_silence=max_silence,
                drop_trailing_silence=drop_trailing_silence,
                strict_min_dur=strict_min_dur,
                **kwargs
            )
            regions = list(regions)
            detections = ((reg.meta.start, reg.meta.end) for reg in regions)
            if plot_kwargs is None:
                plot_kwargs = {}
            plot_detections(
                self, self.sr, detections, show=show, **plot_kwargs
            )
            return regions
        else:
            raise RuntimeWarning("Plotting requires matplotlib")

    def __array__(self):
        return self.samples

    @property
    def samples(self):
        if self._samples is None:
            fmt = signal.FORMAT[self.sample_width]
            if self.channels == 1:
                self._samples = signal.to_array(self._data, fmt)
            else:
                self._samples = signal.separate_channels(
                    self._data, fmt, self.channels
                )
        return self._samples

    def __len__(self):
        """
        Return region length in number of samples.
        """
        return len(self._data) // (self.sample_width * self.channels)

    @property
    def len(self):
        """
        Return region length in number of samples.
        """
        return len(self)

    def __bytes__(self):
        return self._data

    def __str__(self):
        return (
            "AudioRegion(duration={:.3f}, "
            "sampling_rate={}, sample_width={}, channels={})".format(
                self.duration, self.sr, self.sw, self.ch
            )
        )

    def __repr__(self):
        return str(self)

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
        return AudioRegion(data, self.sr, self.sw, self.ch)

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

    def __mul__(self, n):
        if not isinstance(n, int):
            err_msg = "Can't multiply AudioRegion by a non-int of type '{}'"
            raise TypeError(err_msg.format(type(n)))
        data = self._data * n
        return AudioRegion(data, self.sr, self.sw, self.ch)

    def __rmul__(self, n):
        return self * n

    def __truediv__(self, n):
        if not isinstance(n, int) or n <= 0:
            raise TypeError(
                "AudioRegion can only be divided by a positive int"
            )
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
            (self._data == other._data)
            and (self.sr == other.sr)
            and (self.sw == other.sw)
            and (self.ch == other.ch)
        )

    def __getitem__(self, index):
        err_msg = "Slicing AudioRegion by samples requires indices of type "
        err_msg += "'int' without a step (e.g. region.sec[1600:3200])"
        start_sample, stop_sample = _check_convert_index(index, (int), err_msg)

        bytes_per_sample = self.sample_width * self.channels
        len_samples = len(self._data) // bytes_per_sample

        if start_sample < 0:
            start_sample = max(start_sample + len_samples, 0)
        onset = start_sample * bytes_per_sample

        if stop_sample is not None:
            if stop_sample < 0:
                stop_sample = max(stop_sample + len_samples, 0)
            offset = index.stop * bytes_per_sample
        else:
            offset = None

        data = self._data[onset:offset]
        return AudioRegion(data, self.sr, self.sw, self.ch)


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
