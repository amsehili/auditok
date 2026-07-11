"""
Module for pluggable frame validators used by :func:`auditok.split` (via
the ``validator`` parameter) and :class:`auditok.StreamTokenizer`.

.. autosummary::
    :toctree: generated/

    WebRTCVADValidator
"""

from __future__ import annotations

import numpy as np

from .signal import to_array
from .util import DataValidator

__all__ = ["WebRTCVADValidator"]

_WEBRTC_SAMPLING_RATES = (8000, 16000, 32000, 48000)
_WEBRTC_SUBFRAME_DURATIONS = (0.01, 0.02, 0.03)
_AGGREGATIONS = ("majority", "any", "all")


class WebRTCVADValidator(DataValidator):
    """
    A frame validator backed by the WebRTC voice activity detector.

    Unlike :class:`auditok.util.AudioEnergyValidator`, which accepts any
    audio with sufficient energy, this validator only accepts windows that
    the WebRTC VAD — a GMM model over six frequency sub-band energies,
    with online background-noise adaptation — classifies as *speech*.
    Combined with :class:`auditok.StreamTokenizer`'s event machinery
    (``min_dur``, ``max_silence``, leading/trailing silence handling), it
    turns webrtc's frame-level decisions into speech events.

    Requires the optional ``webrtcvad-wheels`` dependency::

        pip install auditok[webrtcvad]

    For the common cases there is a string shortcut on :func:`auditok.split`
    (and ``trim``/``fix_pauses``): ``validator="webrtc"`` or
    ``validator="webrtc:N"`` with N the aggressiveness mode — equivalent to
    constructing this class with default parameters and that mode.

    Notes
    -----
    - The WebRTC VAD is *stateful*: it adapts its noise model to the audio
      it has seen and applies a short decision hangover. Use a fresh
      validator instance per stream/file; do not share one instance
      across unrelated audio.
    - Each analysis window is split into subframes of `subframe_dur`
      seconds (webrtc only accepts 10, 20 or 30 ms frames) and the
      subframe decisions are combined with `aggregation`. Any samples
      left over after the last whole subframe are ignored; a window
      shorter than one subframe is considered not valid.

    Parameters
    ----------
    sampling_rate : int
        Sampling rate of the audio data. Must be 8000, 16000, 32000 or
        48000 Hz (a WebRTC VAD requirement).
    sample_width : int
        Size in bytes of one audio sample (1, 2 or 4). Audio is converted
        to 16-bit internally when needed (4 means 32-bit float, as
        everywhere in auditok).
    channels : int
        Number of channels of the audio data. Multichannel audio is
        reduced to one channel according to `use_channel` before being
        passed to the WebRTC VAD.
    mode : int, default=1
        WebRTC VAD aggressiveness, 0 to 3. Higher values reject more
        audio as non-speech. 0 or 1 are recommended for far-field or
        noisy audio, 2 for clean close-talk audio.
    subframe_dur : float, default=0.01
        Duration in seconds of the frames passed to the WebRTC VAD: 0.01,
        0.02 or 0.03. The analysis window used with :func:`auditok.split`
        should be a multiple of this value.
    aggregation : {"majority", "any", "all"}, default="majority"
        How to combine subframe decisions into a window decision:
        "majority" (at least half the subframes are speech), "any" or
        "all".
    use_channel : int or {"mix", "avg", "average"}, default="mix"
        Channel used for validation when `channels` > 1: a channel index
        or "mix" to average all channels. The energy validator's
        "any"/None semantics (max over channels) are not supported, as
        they would require one stateful VAD instance per channel.
    """

    def __init__(
        self,
        sampling_rate: int,
        sample_width: int,
        channels: int,
        mode: int = 1,
        subframe_dur: float = 0.01,
        aggregation: str = "majority",
        use_channel: int | str = "mix",
    ) -> None:
        try:
            import webrtcvad
        except ImportError as exc:
            raise ImportError(
                "WebRTCVADValidator requires the 'webrtcvad-wheels' "
                "package. Install it with: pip install auditok[webrtcvad]"
            ) from exc

        if sampling_rate not in _WEBRTC_SAMPLING_RATES:
            raise ValueError(
                "WebRTC VAD requires a sampling rate in "
                f"{_WEBRTC_SAMPLING_RATES}, given: {sampling_rate}"
            )
        if subframe_dur not in _WEBRTC_SUBFRAME_DURATIONS:
            raise ValueError(
                "'subframe_dur' must be one of "
                f"{_WEBRTC_SUBFRAME_DURATIONS} (a WebRTC VAD requirement), "
                f"given: {subframe_dur}"
            )
        if aggregation not in _AGGREGATIONS:
            raise ValueError(
                f"'aggregation' must be one of {_AGGREGATIONS}, given: "
                f"{aggregation!r}"
            )
        if channels > 1 and use_channel in (None, "any"):
            raise ValueError(
                "use_channel=None/'any' is not supported by "
                "WebRTCVADValidator (it would require one stateful VAD "
                "per channel); use a channel index or 'mix'"
            )

        self._vad = webrtcvad.Vad(mode)
        self._sampling_rate = sampling_rate
        self._sample_width = sample_width
        self._channels = channels
        self._use_channel = use_channel
        self._aggregation = aggregation
        self._subframe_bytes = int(subframe_dur * sampling_rate) * 2
        # fast path: already 16-bit mono, no conversion needed
        self._passthrough = sample_width == 2 and channels == 1

    def _to_int16_mono(self, data: bytes) -> bytes:
        """Convert an analysis window to 16-bit mono PCM bytes."""
        if self._passthrough:
            return data
        array = to_array(data, self._sample_width, self._channels)
        if self._channels > 1:
            if isinstance(self._use_channel, int):
                array = array[self._use_channel]
            else:  # "mix" / "avg" / "average"
                array = array.mean(axis=0)
        else:
            array = array[0]
        if self._sample_width == 1:
            array = array * 256.0  # int8 scale -> int16 scale
        # float32 input is already on the int16 scale (see signal.to_array)
        return np.clip(array, -32768, 32767).astype(np.dtype("<i2")).tobytes()

    def is_valid(self, data: bytes) -> bool:
        """
        Return True if the WebRTC VAD classifies `data` as speech,
        according to the configured subframe aggregation.
        """
        pcm = self._to_int16_mono(data)
        n_subframes = len(pcm) // self._subframe_bytes
        if n_subframes == 0:
            return False
        decisions = [
            self._vad.is_speech(
                pcm[i * self._subframe_bytes : (i + 1) * self._subframe_bytes],
                self._sampling_rate,
            )
            for i in range(n_subframes)
        ]
        if self._aggregation == "any":
            return any(decisions)
        if self._aggregation == "all":
            return all(decisions)
        return sum(decisions) * 2 >= n_subframes
