import numpy as np
import pytest

from auditok import signal

# from auditok import signal as signal_
# from auditok import signal


@pytest.fixture
def setup_data():
    return b"012345679ABC"


@pytest.mark.parametrize(
    "sample_width, expected",
    [
        (
            1,
            [[48, 49, 50, 51, 52, 53, 54, 55, 57, 65, 66, 67]],
        ),  # int8_1channel
        (
            1,
            [[48, 50, 52, 54, 57, 66], [49, 51, 53, 55, 65, 67]],
        ),  # int8_2channel
        (
            1,
            [[48, 52, 57], [49, 53, 65], [50, 54, 66], [51, 55, 67]],
        ),  # int8_4channel
        (2, [[12592, 13106, 13620, 14134, 16697, 17218]]),  # int16_1channel
        (2, [[12592, 13620, 16697], [13106, 14134, 17218]]),  # int16_2channel
    ],
    ids=[
        "int8_1channel",
        "int8_2channel",
        "int8_4channel",
        "int16_1channel",
        "int16_2channel",
    ],
)
def test_to_array(setup_data, sample_width, expected):
    data = setup_data
    channels = len(expected)
    expected = np.array(expected)
    result = signal.to_array(data, sample_width, channels)
    assert (result == expected).all()
    assert result.dtype == np.float64
    assert result.shape == expected.shape


@pytest.mark.parametrize(
    "channels",
    [1, 2],
    ids=["float32_1channel", "float32_2channel"],
)
def test_to_array_float32(channels):
    """float32 samples are scaled to the int16 amplitude reference so that
    energy values (and `energy_threshold`) mean the same thing for all
    sample widths."""
    samples = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    data = samples.tobytes()
    result = signal.to_array(data, sample_width=4, channels=channels)
    expected = (samples.astype(np.float64) * signal.FLOAT32_SCALE).reshape(
        channels, -1, order="F"
    )
    assert (result == expected).all()
    assert result.dtype == np.float64


def test_to_array_int16_and_float32_same_energy():
    """The same signal stored as int16 and float32 must yield the same
    energy."""
    int16_samples = (np.sin(np.linspace(0, 8 * np.pi, 160)) * 20000).astype(
        np.int16
    )
    float32_samples = (int16_samples / 32768.0).astype(np.float32)
    energy_int16 = signal.calculate_energy(
        signal.to_array(int16_samples.tobytes(), 2, 1)
    )
    energy_float32 = signal.calculate_energy(
        signal.to_array(float32_samples.tobytes(), 4, 1)
    )
    assert np.isclose(energy_int16, energy_float32, atol=1e-3)


@pytest.mark.parametrize(
    "x, aggregation_fn, expected",
    [
        ([300, 320, 400, 600], None, 52.506639194632434),  # mono_simple
        ([0, 0, 0], None, -200),  # mono_zeros
        (
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            None,
            [52.506639194632434, 46.48603928135281],
        ),  # stereo_no_agg
        (
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            np.mean,
            49.49633923799262,
        ),  # stereo_mean_agg
        (
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            min,
            46.48603928135281,
        ),  # stereo_min_agg
        (
            [[300, 320, 400, 600], [150, 160, 200, 300]],
            max,
            52.506639194632434,
        ),  # stereo_max_agg
    ],
    ids=[
        "mono_simple",
        "mono_zeros",
        "stereo_no_agg",
        "mean_agg",
        "stereo_min_agg",
        "stereo_max_agg",
    ],
)
def test_calculate_energy(x, aggregation_fn, expected):
    energy = signal.calculate_energy(x, aggregation_fn)
    assert (energy == expected).all()


@pytest.mark.parametrize(
    "channels, use_channel",
    [
        (1, None),
        (3, None),
        (3, "any"),
        (3, 0),
        (3, -1),
        (3, "mix"),
    ],
    ids=[
        "mono",
        "multichannel_any_default",
        "multichannel_any_explicit",
        "multichannel_channel_0",
        "multichannel_negative_index",
        "multichannel_mix",
    ],
)
def test_compute_frame_energies_matches_validator(channels, use_channel):
    """The vectorized per-window energies must be exactly what
    `AudioEnergyValidator` computes window by window — this is what makes a
    threshold estimated from them mean the same thing to the tokenizer."""
    from auditok.util import make_channel_selector

    rng = np.random.RandomState(1234)
    sample_width = 2
    samples = (rng.randn(16000 * channels) * 5000).astype(np.int16)
    data = samples.tobytes()
    frame_samples = 777  # not a divisor: final partial window is dropped

    result = signal.compute_frame_energies(
        data, sample_width, channels, frame_samples, use_channel
    )

    selector = make_channel_selector(sample_width, channels, use_channel)
    frame_bytes = frame_samples * sample_width * channels
    expected = [
        float(np.max(signal.calculate_energy(selector(frame))))
        for frame in (
            data[i * frame_bytes : (i + 1) * frame_bytes]
            for i in range(len(data) // frame_bytes)
        )
    ]
    assert result.shape == (len(expected),)
    assert np.allclose(result, expected)


def test_compute_frame_energies_short_input():
    data = np.zeros(10, dtype=np.int16).tobytes()
    result = signal.compute_frame_energies(data, 2, 1, 100)
    assert result.shape == (0,)


def test_estimate_energy_threshold_bimodal():
    """All methods must place the threshold above the noise mode on a
    clearly bimodal energy distribution."""
    rng = np.random.RandomState(0)
    noise = rng.normal(20, 2, 500)
    activity = rng.normal(60, 3, 300)
    energies = np.concatenate([noise, activity])
    for method in ("otsu", "percentile"):
        threshold = signal.estimate_energy_threshold(energies, method=method)
        assert 20 < threshold < 69, f"{method}: {threshold}"
    # otsu must fall between the two modes
    threshold = signal.estimate_energy_threshold(energies, method="otsu")
    assert 28 < threshold < 52, f"otsu: {threshold}"


def test_estimate_energy_threshold_method_args():
    energies = np.arange(100, dtype=np.float64)
    threshold = signal.estimate_energy_threshold(
        energies, method="percentile", percentile=50.0, margin=3.0
    )
    assert threshold == pytest.approx(np.percentile(energies, 50) + 3.0)


def test_estimate_energy_threshold_ignores_digital_silence():
    """All-zero windows are clipped to -200 dB by `calculate_energy`; that
    sentinel must not influence the estimate (regression: digitally silent
    padding dragged all estimators far below the real noise floor)."""
    rng = np.random.RandomState(0)
    noise = rng.normal(20, 2, 500)
    activity = rng.normal(60, 3, 300)
    energies = np.concatenate([noise, activity])
    digital_silence = np.full(400, 20 * np.log10(signal.EPSILON))
    padded = np.concatenate([digital_silence, energies])
    for method in ("otsu", "percentile"):
        assert signal.estimate_energy_threshold(
            padded, method=method
        ) == signal.estimate_energy_threshold(energies, method=method)


def test_estimate_energy_threshold_degenerate_input():
    # constant energies: nothing to separate, return the single value
    assert signal.estimate_energy_threshold([42.0] * 10) == 42.0
    # entirely digitally silent: there is nothing to detect, so no
    # finite threshold — every window must fail validation
    silence = 20 * np.log10(signal.EPSILON)
    assert signal.estimate_energy_threshold([silence] * 10) == float("inf")
    with pytest.raises(ValueError):
        signal.estimate_energy_threshold([])
    with pytest.raises(ValueError):
        signal.estimate_energy_threshold([1.0, 2.0], method="unknown")
