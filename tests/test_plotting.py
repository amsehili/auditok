import os
import sys
import unittest
from unittest import TestCase
from tempfile import TemporaryDirectory
from genty import genty, genty_dataset
import matplotlib

matplotlib.use("AGG")  # noqa E402
import matplotlib.pyplot as plt
from auditok.core import AudioRegion

if sys.version_info.minor <= 5:
    PREFIX = "py34_py35/"
else:
    PREFIX = ""

matplotlib.rcParams["figure.figsize"] = (10, 4)


@genty
class TestPlotting(TestCase):
    @genty_dataset(mono=(1,), stereo=(2,))
    def test_region_plot(self, channels):
        type_ = "mono" if channels == 1 else "stereo"
        audio_filename = "tests/data/test_split_10HZ_{}.raw".format(type_)
        image_filename = "tests/images/{}plot_{}_region.png".format(
            PREFIX, type_
        )
        expected_image = plt.imread(image_filename)
        with TemporaryDirectory() as tmpdir:
            output_image_filename = os.path.join(tmpdir, "image.png")
            region = AudioRegion.load(audio_filename, sr=10, sw=2, ch=channels)
            region.plot(show=False, save_as=output_image_filename)
            output_image = plt.imread(output_image_filename)
        self.assertTrue((output_image == expected_image).all())

    @genty_dataset(
        mono=(1,),
        stereo_any=(2, "any"),
        stereo_uc_0=(2, 0),
        stereo_uc_1=(2, 1),
        stereo_uc_mix=(2, "mix"),
    )
    def test_region_split_and_plot(self, channels, use_channel=None):
        type_ = "mono" if channels == 1 else "stereo"
        audio_filename = "tests/data/test_split_10HZ_{}.raw".format(type_)
        if type_ == "mono":
            fmt = "tests/images/{}split_and_plot_mono_region.png"
        else:
            fmt = "tests/images/{}split_and_plot_uc_{}_stereo_region.png"
        image_filename = fmt.format(PREFIX, use_channel)

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
        self.assertTrue((output_image == expected_image).all())


if __name__ == "__main__":
    unittest.main()
