from unittest import TestCase
import math
from array import array
from genty import genty, genty_dataset
from auditok.util import AudioEnergyValidator
from auditok.io import DATA_FORMAT


def _sample_generator(*data_buffers):
    """
    Takes a list of many mono audio data buffers and makes a sample generator
    of interleaved audio samples, one sample from each channel. The resulting
    generator can be used to build a multichannel audio buffer.
    >>> gen = _sample_generator("abcd", "ABCD")
    >>> list(gen)
    ["a", "A", "b", "B", "c", "C", "d", "D"]
    """
    frame_gen = zip(*data_buffers)
    return (sample for frame in frame_gen for sample in frame)


def _generate_pure_tone(
    frequency, duration_sec=1, sampling_rate=16000, sample_width=2, volume=1e4
):
    """
    Generates a pure tone with the given frequency.
    """
    assert frequency <= sampling_rate / 2
    max_value = (2 ** (sample_width * 8) // 2) - 1
    if volume > max_value:
        volume = max_value
    fmt = DATA_FORMAT[sample_width]
    total_samples = int(sampling_rate * duration_sec)
    step = frequency / sampling_rate
    two_pi_step = 2 * math.pi * step
    data = array(
        fmt,
        (
            int(math.sin(two_pi_step * i) * volume)
            for i in range(total_samples)
        ),
    )
    return data


PURE_TONE_DICT = {
    freq: _generate_pure_tone(freq, 1, 16000, 2) for freq in (400, 800, 1600)
}
PURE_TONE_DICT.update(
    {
        freq: _generate_pure_tone(freq, 0.1, 16000, 2)
        for freq in (600, 1150, 2400, 7220)
    }
)


@genty
class TestAudioEnergyValidator(TestCase):
    @genty_dataset(
        mono_valid_uc_None=([350, 400], 1, None, True),
        mono_valid_uc_any=([350, 400], 1, "any", True),
        mono_valid_uc_0=([350, 400], 1, 0, True),
        mono_valid_uc_mix=([350, 400], 1, "mix", True),
        # previous cases are all the same since we have mono audio
        mono_invalid_uc_None=([300, 300], 1, None, False),
        stereo_valid_uc_None=([300, 400, 350, 300], 2, None, True),
        stereo_valid_uc_any=([300, 400, 350, 300], 2, "any", True),
        stereo_valid_uc_mix=([300, 400, 350, 300], 2, "mix", True),
        stereo_valid_uc_avg=([300, 400, 350, 300], 2, "avg", True),
        stereo_valid_uc_average=([300, 400, 300, 300], 2, "average", True),
        stereo_valid_uc_mix_with_null_channel=(
            [634, 0, 634, 0],
            2,
            "mix",
            True,
        ),
        stereo_valid_uc_0=([320, 100, 320, 100], 2, 0, True),
        stereo_valid_uc_1=([100, 320, 100, 320], 2, 1, True),
        stereo_invalid_uc_None=([280, 100, 280, 100], 2, None, False),
        stereo_invalid_uc_any=([280, 100, 280, 100], 2, "any", False),
        stereo_invalid_uc_mix=([400, 200, 400, 200], 2, "mix", False),
        stereo_invalid_uc_0=([300, 400, 300, 400], 2, 0, False),
        stereo_invalid_uc_1=([400, 300, 400, 300], 2, 1, False),
        zeros=([0, 0, 0, 0], 2, None, False),
    )
    def test_audio_energy_validator(
        self, data, channels, use_channel, expected
    ):

        data = array("h", data)
        sample_width = 2
        energy_threshold = 50
        validator = AudioEnergyValidator(
            energy_threshold, sample_width, channels, use_channel
        )

        if expected:
            self.assertTrue(validator.is_valid(data))
        else:
            self.assertFalse(validator.is_valid(data))
