"""
Module for main signal processing operations.

.. autosummary::
    :toctree: generated/

    to_array
    calculate_energy
    compute_frame_energies
    estimate_energy_threshold
"""

import numpy as np

__all__ = [
    "SAMPLE_WIDTH_TO_DTYPE",
    "to_array",
    "calculate_energy",
    "compute_frame_energies",
    "estimate_energy_threshold",
]

SAMPLE_WIDTH_TO_DTYPE = {1: np.int8, 2: np.int16, 4: np.float32}
EPSILON = 1e-10

# Full scale of 16-bit audio, used as the common amplitude reference for all
# sample widths. float32 samples (nominally in [-1.0, 1.0]) are scaled by
# this factor in `to_array` so that energy values and `energy_threshold`
# keep the same meaning regardless of the storage format.
FLOAT32_SCALE = 32768.0


def _get_numpy_dtype(sample_width):
    """
    Helper function to convert a sample width to the corresponding NumPy data
    type.

    Parameters
    ----------
    sample_width : int
        The width of the sample in bytes. Accepted values are 1, 2 or 4.
        A width of 4 means 32-bit IEEE float samples (not 32-bit integers).

    Returns
    -------
    numpy.dtype
        The corresponding NumPy data type for the specified sample width.

    Raises
    ------
    ValueError
        If `sample_width` is not one of the accepted values (1, 2 or 4).
    """

    dtype = SAMPLE_WIDTH_TO_DTYPE.get(sample_width)
    if dtype is None:
        err_msg = "'sample_width' must be 1, 2 or 4, given: {}"
        raise ValueError(err_msg.format(sample_width))
    return dtype


def to_array(data, sample_width, channels):
    """
    Convert raw audio data into a NumPy array.

    This function transforms raw audio data, specified by sample width and
    number of channels, into a 2-D NumPy array of `numpy.float64` data type.
    The array will be arranged by channels and samples.

    Samples keep their integer scale: 16-bit audio yields values in
    [-32768, 32767]. 32-bit float samples (`sample_width=4`, nominally in
    [-1.0, 1.0]) are scaled by 32768 so that amplitudes—and therefore
    energy values and `energy_threshold`—mean the same thing for all
    sample widths.

    Parameters
    ----------
    data : bytes
        The raw audio data.
    sample_width : int
        The sample width (in bytes) of each audio sample.
    channels : int
        The number of audio channels.

    Returns
    -------
    numpy.ndarray
        A 2-D NumPy array representing the audio data. The shape of the array
        will be (number of channels, number of samples), with data type
        `numpy.float64`.

    Raises
    ------
    ValueError
        If `sample_width` is not an accepted value for conversion by the helper
        function `_get_numpy_dtype`.
    """

    dtype = _get_numpy_dtype(sample_width)
    array = np.frombuffer(data, dtype=dtype).astype(np.float64)
    if sample_width == 4:
        array *= FLOAT32_SCALE
    return array.reshape(channels, -1, order="F")


def calculate_energy(x, agg_fn=None):
    """Calculate the energy of audio data.

    The energy is calculated as:

    .. math::
       \\text{energy} = 20 \\log(\\sqrt({1}/{N} \\sum_{i=1}^{N} {a_i}^2))  % # noqa: W605

    where `a_i` is the i-th audio sample and `N` is the total number of samples
    in `x`. Samples are expected on the int16 amplitude scale (as returned by
    `to_array` for all sample widths), so energy values are comparable across
    storage formats.

    Parameters
    ----------
    x : array
        Array of audio data, which may contain multiple channels.
    agg_fn : callable, optional
        Aggregation function to use for multi-channel data. If None, the energy
        will be computed and returned separately for each channel.

    Returns
    -------
    float or numpy.ndarray
        The energy of the audio signal. If `x` is multichannel and `agg_fn` is
        None, this will be an array of energies, one per channel.
    """

    x = np.array(x).astype(np.float64)
    energy_sqrt = np.sqrt(np.mean(x**2, axis=-1))
    energy_sqrt = np.clip(energy_sqrt, a_min=EPSILON, a_max=None)
    energy = 20 * np.log10(energy_sqrt)
    if agg_fn is not None:
        energy = agg_fn(energy)
    return energy


def compute_frame_energies(
    data, sample_width, channels, frame_samples, use_channel=None
):
    """Compute the energy of each analysis window of `data`, vectorized.

    The energy formula and channel aggregation are exactly those of
    :class:`auditok.util.AudioEnergyValidator`, so a threshold estimated
    from the returned energies means the same thing to the tokenizer as a
    user-provided ``energy_threshold``. A final partial window, if any, is
    ignored.

    Parameters
    ----------
    data : bytes
        Raw audio data.
    sample_width : int
        The sample width (in bytes) of each audio sample.
    channels : int
        The number of audio channels.
    frame_samples : int
        Number of samples per analysis window.
    use_channel : {None, "any", "mix", "avg", "average"} or int
        Channel used for energy computation, with the same semantics as in
        :class:`auditok.util.AudioEnergyValidator`.

    Returns
    -------
    numpy.ndarray
        1-D array of window energies, of length
        ``len(samples) // frame_samples``.
    """

    arr = to_array(data, sample_width, channels)
    n_frames = arr.shape[-1] // frame_samples
    if n_frames == 0:
        return np.empty(0, dtype=np.float64)
    arr = arr[..., : n_frames * frame_samples]
    arr = arr.reshape(channels, n_frames, frame_samples)
    if channels == 1:
        return calculate_energy(arr[0])
    if use_channel in (None, "any"):
        return calculate_energy(arr).max(axis=0)
    if isinstance(use_channel, int) and not isinstance(use_channel, bool):
        selected = use_channel + channels if use_channel < 0 else use_channel
        if selected < 0 or selected >= channels:
            err_msg = "Selected channel must be >= -channels and < channels"
            err_msg += ", given: {}"
            raise ValueError(err_msg.format(use_channel))
        return calculate_energy(arr[selected])
    if use_channel in ("mix", "avg", "average"):
        return calculate_energy(arr.mean(axis=0))
    raise ValueError(
        "Selected channel must be an integer, None (alias 'any') or "
        "'average' (alias 'avg' or 'mix')"
    )


def _estimate_threshold_otsu(energies, bins=128):
    """Otsu's method: split the energy histogram in two classes maximizing
    the between-class variance. Parameter-free; assumes the energy
    distribution is roughly bimodal (background vs. activity)."""
    hist, edges = np.histogram(energies, bins=bins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_0 = np.cumsum(hist)[:-1]  # class 0 = bins [0..i]
    weight_1 = hist.sum() - weight_0
    cum_mass = np.cumsum(hist * centers)[:-1]
    total_mass = (hist * centers).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_0 = cum_mass / weight_0
        mu_1 = (total_mass - cum_mass) / weight_1
        between_var = weight_0 * weight_1 * (mu_0 - mu_1) ** 2
    between_var = np.nan_to_num(between_var, nan=-1.0)
    # empty bins between the two modes make the between-class variance
    # exactly flat over the gap; take the middle of the plateau (max
    # margin) rather than argmax's leftmost point
    candidates = np.flatnonzero(between_var == between_var.max())
    split = candidates[(len(candidates) - 1) // 2]
    return edges[split + 1]


def _estimate_threshold_percentile(energies, percentile=10.0, margin=6.0):
    """Noise floor (low percentile of window energies) plus a margin."""
    return np.percentile(energies, percentile) + margin


_THRESHOLD_ESTIMATORS = {
    "otsu": _estimate_threshold_otsu,
    "percentile": _estimate_threshold_percentile,
}

DEFAULT_THRESHOLD_METHOD = "otsu"


def estimate_energy_threshold(
    frame_energies, method=DEFAULT_THRESHOLD_METHOD, **method_args
):
    """Estimate an energy threshold from analysis-window energies.

    This is what automatic thresholding (``validator="otsu"``,
    ``"percentile"`` or ``"pXX"``) uses under the hood (with the default
    method). Energies are typically obtained with
    :func:`compute_frame_energies`; the returned value is on the same dB
    scale as ``energy_threshold`` and can be passed to :func:`split`
    directly.

    Parameters
    ----------
    frame_energies : array-like
        Energies of the analysis windows of the audio signal.
    method : str, default="otsu"
        Estimation method. One of:

        - "otsu": split the energy histogram in two classes, maximizing
          between-class variance (parameter-free). A balanced choice,
          well suited to audio with clear pauses.
        - "percentile": noise floor estimated as a low percentile of
          window energies, plus a margin. Accepts ``percentile`` (default
          10.0) and ``margin`` (default 6.0) keyword arguments. More
          recall-oriented; a good choice for dense or far-field speech.
    method_args
        Optional method-specific arguments, see above.

    Returns
    -------
    float
        The estimated energy threshold.

    Raises
    ------
    ValueError
        If `frame_energies` is empty or `method` is unknown.
    """

    energies = np.asarray(frame_energies, dtype=np.float64).ravel()
    if energies.size == 0:
        raise ValueError(
            "Cannot estimate an energy threshold from an empty energy "
            "array (input audio shorter than one analysis window?)"
        )
    # `calculate_energy` clips all-zero (digitally silent) windows to
    # 20*log10(EPSILON) = -200 dB. That value is a sentinel, not a
    # measurement; keeping it would create an artificial low mode that
    # drags every estimator far below the real noise floor (e.g., in
    # studio recordings with digitally silent edits or padding).
    silence_sentinel = 20 * np.log10(EPSILON)
    non_silent = energies[energies > silence_sentinel]
    if non_silent.size > 0:
        energies = non_silent
    if energies.min() == energies.max():
        return float(energies[0])
    estimator = _THRESHOLD_ESTIMATORS.get(method)
    if estimator is None:
        raise ValueError(
            "Unknown threshold estimation method {!r}, expected one of "
            "{}".format(method, sorted(_THRESHOLD_ESTIMATORS))
        )
    return float(estimator(energies, **method_args))
