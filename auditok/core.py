"""
Module for low-level tokenization algorithms.

.. autosummary::
    :toctree: generated/

    StreamTokenizer
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Generator

from .util import DataSource, DataValidator

__all__ = [
    "StreamTokenizer",
]


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


# ---------------------------------------------------------------------------
# Backward-compatible re-exports from auditok.audio
#
# Names that used to live in core.py (AudioRegion, split, load, etc.) are
# now in audio.py.  The __getattr__ below makes ``from auditok.core import
# AudioRegion`` (and similar) keep working without eagerly importing
# audio.py at module level, which would create a circular import.
# ---------------------------------------------------------------------------


def __getattr__(name):
    from . import audio as _audio

    try:
        return getattr(_audio, name)
    except AttributeError:
        raise AttributeError(
            f"module 'auditok.core' has no attribute {name!r}"
        ) from None
