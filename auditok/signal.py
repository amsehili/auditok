"""
Module for main signal processing operations.

.. autosummary::
    :toctree: generated/

    to_array
    calculate_energy
"""

import numpy as np

__all__ = [
    "SAMPLE_WIDTH_TO_DTYPE",
    "to_array",
    "calculate_energy",
]

SAMPLE_WIDTH_TO_DTYPE = {1: np.int8, 2: np.int16, 4: np.int32}
EPSILON = 1e-10


def _get_numpy_dtype(sample_width):
    """
    Helper function to convert a sample width to the corresponding NumPy data
    type.

    Parameters
    ----------
    sample_width : int
        The width of the sample in bytes. Accepted values are 1, 2, or 4.

    Returns
    -------
    numpy.dtype
        The corresponding NumPy data type for the specified sample width.

    Raises
    ------
    ValueError
        If `sample_width` is not one of the accepted values (1, 2, or 4).
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
    return array.reshape(channels, -1, order="F")


def calculate_energy(x, agg_fn=None):
    """Calculate the energy of audio data.

    The energy is calculated as:

    .. math::
       \\text{energy} = 20 \\log(\\sqrt({1}/{N} \\sum_{i=1}^{N} {a_i}^2))  % # noqa: W605

    where `a_i` is the i-th audio sample and `N` is the total number of samples
    in `x`.

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
