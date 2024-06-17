import numpy as np

SAMPLE_WIDTH_TO_DTYPE = {1: np.int8, 2: np.int16, 4: np.int32}
EPSILON = 1e-10


def _get_numpy_dtype(sample_width):
    """Helper function to convert sample with to the corresponding numpy type."""
    dtype = SAMPLE_WIDTH_TO_DTYPE.get(sample_width)
    if dtype is None:
        err_msg = "'sample_width' must be 1, 2 or 4, given: {}"
        raise ValueError(err_msg.format(sample_width))
    return dtype


def to_array(data, sample_width, channels):
    """
    Convert raw audio data into a NumPy array.

    The returned array will have a data type of `numpy.float64` regardless of
    the sample width.

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
        A 2-D NumPy array representing the audio data. The array will have a
        shape of (number of channels, number of samples) and will be of data
        type `numpy.float64`.
    """
    dtype = _get_numpy_dtype(sample_width)
    array = np.frombuffer(data, dtype=dtype).astype(np.float64)
    return array.reshape(channels, -1, order="F")


def calculate_energy(x, agg_fn=None):
    """Calculate the energy of audio data. The energy is calculated as:

    .. math:: energy = 20 \log(\sqrt({1}/{N}\sum_{i}^{N}{a_i}^2)) % # noqa: W605

    where `a_i` is the i-th audio sample and `N` is the number of audio samples
    in data.

    Parameters
    ----------
    x : array
        array of audio data.
    agg_fn : callable
        aggregation function to use for multi-channel data. If None, the energy
        will be computed and returned for each channel separately.


    Returns
    -------
    energy : float, numpy.ndarray
        energy of audio signal. If x is multichannel and agg_fn is None, this
        an array of energies, one per channel.
    """
    x = np.array(x).astype(np.float64)
    energy_sqrt = np.sqrt(np.mean(x**2, axis=-1))
    energy_sqrt = np.clip(energy_sqrt, a_min=EPSILON, a_max=None)
    energy = 20 * np.log10(energy_sqrt)
    if agg_fn is not None:
        energy = agg_fn(energy)
    return energy
