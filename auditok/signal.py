from array import array
import math

FORMAT = {1: "b", 2: "h", 4: "i"}
_EPSILON = 1e-20


def to_array(data, fmt):
    return array(fmt, data)


def extract_single_channel(data, fmt, channels, selected):
    samples = array(fmt, data)
    return samples[selected::channels]


def average_channels(data, fmt, channels):
    all_channels = array(fmt, data)
    mono_channels = [
        array(fmt, all_channels[ch::channels]) for ch in range(channels)
    ]
    avg_arr = array(
        fmt,
        (round(sum(samples) / channels) for samples in zip(*mono_channels)),
    )
    return avg_arr


def separate_channels(data, fmt, channels):
    all_channels = array(fmt, data)
    mono_channels = [
        array(fmt, all_channels[ch::channels]) for ch in range(channels)
    ]
    return mono_channels


def calculate_energy_single_channel(x):
    energy = max(sum(i ** 2 for i in x) / len(x), _EPSILON)
    return 10 * math.log10(energy)


def calculate_energy_multichannel(x, aggregation_fn=max):
    energies = (calculate_energy_single_channel(xi) for xi in x)
    return aggregation_fn(energies)
