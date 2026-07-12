"""Tests for auditok.plotting.

These tests assert on the *contents* of the matplotlib figure (axes,
line data, detection spans, threshold line) rather than comparing
rendered images pixel by pixel: rasterization details (fonts,
antialiasing) change across matplotlib/numpy versions, while the
plotted data — what auditok actually controls — must not.
"""

import os
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pytest

matplotlib.use("AGG")
import matplotlib.pyplot as plt  # noqa E402

from auditok import split  # noqa E402
from auditok.audio import AudioRegion  # noqa E402

matplotlib.rcParams["figure.figsize"] = (10, 4)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _scaled(samples):
    """The scaling applied by plot(): per-channel standardization."""
    std = samples.std()
    if std > 0:
        return (samples - samples.mean()) / std
    return samples


def _scaled_threshold(samples, energy_threshold):
    """The plotted amplitude of the energy threshold on a scaled axis."""
    amplitude = 10 ** (energy_threshold / 20)
    return (amplitude - samples.mean()) / samples.std()


def _signal_lines(ax):
    """Signal line(s) of an axis, excluding the threshold line."""
    return [
        line for line in ax.lines if line.get_label() != "Detection threshold"
    ]


def _span_ranges(ax):
    """(start, end) time ranges of the axvspan detection patches.

    axvspan returns a Polygon on older matplotlib versions and a
    Rectangle on newer ones."""
    ranges = []
    for patch in ax.patches:
        if hasattr(patch, "get_width"):  # Rectangle
            start = patch.get_x()
            ranges.append((start, start + patch.get_width()))
        else:  # Polygon
            xs = patch.get_xy()[:, 0]
            ranges.append((xs.min(), xs.max()))
    return sorted(ranges)


@pytest.mark.parametrize("channels", [1, 2], ids=["mono", "stereo"])
def test_region_plot(channels):
    type_ = "mono" if channels == 1 else "stereo"
    audio_filename = "tests/data/test_split_10HZ_{}.raw".format(type_)
    region = AudioRegion.load(audio_filename, sr=10, sw=2, ch=channels)

    with TemporaryDirectory() as tmpdir:
        output_image_filename = os.path.join(tmpdir, "image.png")
        region.plot(show=False, save_as=output_image_filename)
        # a non-empty, readable PNG must be written
        image = plt.imread(output_image_filename)
        assert image.size > 0

    fig = plt.gcf()
    assert len(fig.axes) == channels
    expected = np.asarray(region).reshape(channels, -1)
    n_samples = expected.shape[1]
    for i, ax in enumerate(fig.axes):
        assert ax.get_title() == f"Channel {i + 1}"
        (line,) = _signal_lines(ax)
        data = line.get_xydata()
        # time axis: n_samples points at 1/sr spacing
        assert data.shape[0] == n_samples
        assert np.allclose(data[:, 0], np.arange(n_samples) / 10)
        # plotted signal: per-channel standardized samples
        assert np.allclose(data[:, 1], _scaled(expected[i]))
        # no detections were passed: no spans
        assert len(ax.patches) == 0


@pytest.mark.parametrize(
    "channels, use_channel",
    [
        (1, None),  # mono
        (2, "any"),  # stereo_any
        (2, 0),  # stereo_uc_0
        (2, 1),  # stereo_uc_1
        (2, "mix"),  # stereo_uc_mix
    ],
    ids=["mono", "stereo_any", "stereo_uc_0", "stereo_uc_1", "stereo_uc_mix"],
)
def test_region_split_and_plot(channels, use_channel):
    type_ = "mono" if channels == 1 else "stereo"
    audio_filename = "tests/data/test_split_10HZ_{}.raw".format(type_)
    region = AudioRegion.load(audio_filename, sr=10, sw=2, ch=channels)

    with TemporaryDirectory() as tmpdir:
        output_image_filename = os.path.join(tmpdir, "image.png")
        region.split_and_plot(
            aw=0.1,
            uc=use_channel,
            max_silence=0,
            show=False,
            save_as=output_image_filename,
        )
        image = plt.imread(output_image_filename)
        assert image.size > 0

    # reference detections from split() with identical parameters
    expected_events = [
        (r.start, r.end)
        for r in split(region, aw=0.1, uc=use_channel, max_silence=0)
    ]
    assert len(expected_events) > 0
    # plot() draws span ends at the start of the last sample
    expected_spans = sorted(
        (start, end - 1 / 10) for start, end in expected_events
    )

    fig = plt.gcf()
    assert len(fig.axes) == channels
    expected = np.asarray(region).reshape(channels, -1)
    for i, ax in enumerate(fig.axes):
        # signal line, standardized
        (line,) = _signal_lines(ax)
        assert np.allclose(line.get_ydata(), _scaled(expected[i]))
        # detection spans on every channel, at the detected time ranges
        assert np.allclose(_span_ranges(ax), expected_spans)
        # energy threshold line: horizontal, at the scaled default (50 dB)
        threshold_lines = [
            line
            for line in ax.lines
            if line.get_label() == "Detection threshold"
        ]
        assert len(threshold_lines) == 1
        threshold_y = threshold_lines[0].get_ydata()
        assert threshold_y[0] == threshold_y[-1]
        assert np.allclose(threshold_y[0], _scaled_threshold(expected[i], 50))
