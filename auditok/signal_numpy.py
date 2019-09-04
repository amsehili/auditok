import numpy as np

FORMAT = {1: np.int8, 2: np.int16, 4: np.int32}
_EPSILON = 1e-20


def to_array(data, fmt):
    return np.frombuffer(data, dtype=fmt).astype(np.float64)


def extract_single_channel(data, fmt, channels, selected):
    samples = np.frombuffer(data, dtype=fmt)
    return samples[selected::channels].astype(np.float64)


def average_channels(data, fmt, channels):
    array = np.frombuffer(data, dtype=fmt).astype(np.float64)
    return array.reshape(-1, channels).mean(axis=1).round()


def separate_channels(data, fmt, channels):
    array = np.frombuffer(data, dtype=fmt).astype(np.float64)
    return array.reshape(-1, channels).T


def calculate_energy_single_channel(x):
    x = np.asarray(x)
    return 10 * np.log10((np.dot(x, x) / x.size).clip(min=_EPSILON))


def calculate_energy_multichannel(x, aggregation_fn=np.max):
    x = np.asarray(x)
    energy = 10 * np.log10((x * x).mean(axis=1).clip(min=_EPSILON))
    return aggregation_fn(energy)
