import os
import sys
from tempfile import TemporaryDirectory

import matplotlib
import pytest

matplotlib.use("AGG")
import matplotlib.pyplot as plt  # noqa E402

from auditok.core import AudioRegion  # noqa E402

SAVE_NEW_IMAGES = False
if SAVE_NEW_IMAGES:
    import shutil  # noqa E402

matplotlib.rcParams["figure.figsize"] = (10, 4)


@pytest.mark.parametrize("channels", [1, 2], ids=["mono", "stereo"])
def test_region_plot(channels):
    type_ = "mono" if channels == 1 else "stereo"
    audio_filename = "tests/data/test_split_10HZ_{}.raw".format(type_)
    image_filename = "tests/images/plot_{}_region.png".format(type_)
    expected_image = plt.imread(image_filename)
    with TemporaryDirectory() as tmpdir:
        output_image_filename = os.path.join(tmpdir, "image.png")
        region = AudioRegion.load(audio_filename, sr=10, sw=2, ch=channels)
        region.plot(show=False, save_as=output_image_filename)
        output_image = plt.imread(output_image_filename)

        if SAVE_NEW_IMAGES:
            shutil.copy(output_image_filename, image_filename)
    assert (output_image == expected_image).all()  # mono, stereo


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
    if type_ == "mono":
        image_filename = "tests/images/split_and_plot_mono_region.png"
    else:
        image_filename = (
            f"tests/images/split_and_plot_uc_{use_channel}_stereo_region.png"
        )

    expected_image = plt.imread(image_filename)
    with TemporaryDirectory() as tmpdir:
        output_image_filename = os.path.join(tmpdir, "image.png")
        region = AudioRegion.load(audio_filename, sr=10, sw=2, ch=channels)
        region.split_and_plot(
            aw=0.1,
            uc=use_channel,
            max_silence=0,
            show=False,
            save_as=output_image_filename,
        )
        output_image = plt.imread(output_image_filename)

        if SAVE_NEW_IMAGES:
            shutil.copy(output_image_filename, image_filename)
    assert (output_image == expected_image).all()
