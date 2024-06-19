import sys
import wave
from functools import partial

import pytest

from auditok import (
    AudioReader,
    BufferAudioSource,
    Recorder,
    WaveAudioSource,
    dataset,
)
from auditok.util import _Limiter, _OverlapAudioReader


def _read_all_data(reader):
    blocks = []
    while True:
        data = reader.read()
        if data is None:
            break
        blocks.append(data)
    return b"".join(blocks)


class TestAudioReaderWithFileAudioSource:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        self.audio_source.open()
        yield
        self.audio_source.close()

    def test_AudioReader_type(self):
        reader = AudioReader(input=self.audio_source)
        err_msg = "wrong object type, expected: 'AudioReader', found: {0}"
        assert isinstance(reader, AudioReader), err_msg.format(type(reader))

    def _test_default_block_size(self):
        reader = AudioReader(input=self.audio_source)
        data = reader.read()
        size = len(data)
        assert (
            size == 160
        ), "Wrong default block_size, expected: 160, found: {0}".format(size)

    @pytest.mark.parametrize(
        "block_dur, expected_nb_samples",
        [
            (None, 160),  # default: 10 ms
            (0.025, 400),  # 25 ms
        ],
        ids=["default", "_25ms"],
    )
    def test_block_duration(self, block_dur, expected_nb_samples):
        """Test the number of samples read for a given block duration."""
        if block_dur is not None:
            reader = AudioReader(input=self.audio_source, block_dur=block_dur)
        else:
            reader = AudioReader(input=self.audio_source)
        data = reader.read()
        nb_samples = len(data) // reader.sample_width
        assert (
            nb_samples == expected_nb_samples
        ), f"Wrong block_size, expected: {expected_nb_samples}, found: {nb_samples}"

    @pytest.mark.parametrize(
        "block_dur, hop_dur, expected_nb_blocks, expected_last_block_nb_samples",
        [
            (None, None, 1879, 126),  # default: 10 ms
            (0.01, None, 1879, 126),  # block_dur_10ms_hop_dur_None
            (0.01, 0.01, 1879, 126),  # block_dur_10ms_hop_dur_10ms
            (0.02, None, 940, 126),  # block_dur_20ms_hop_dur_None
            (0.025, None, 752, 206),  # block_dur_25ms_hop_dur_None
            (0.02, 0.01, 1878, 286),  # block_dur_20ms_hop_dur_10ms
            (0.025, 0.005, 3754, 366),  # block_dur_25ms_hop_dur_5ms
        ],
        ids=[
            "default",
            "block_dur_10ms_hop_dur_None",
            "block_dur_10ms_hop_dur_10ms",
            "block_dur_20ms_hop_dur_None",
            "block_dur_25ms_hop_dur_None",
            "block_dur_20ms_hop_dur_10ms",
            "block_dur_25ms_hop_dur_5ms",
        ],
    )
    def test_hop_duration(
        self,
        block_dur,
        hop_dur,
        expected_nb_blocks,
        expected_last_block_nb_samples,
    ):
        """Test the number of read blocks and the duration of last block for
        different 'block_dur' and 'hop_dur' values.

        Args:
            block_dur (float or None): block duration in seconds.
            hop_dur (float or None): hop duration in seconds.
            expected_nb_blocks (int): expected number of read block.
            expected_last_block_nb_samples (int): expected number of sample
                in the last block.
        """
        if block_dur is not None:
            reader = AudioReader(
                input=self.audio_source, block_dur=block_dur, hop_dur=hop_dur
            )
        else:
            reader = AudioReader(input=self.audio_source, hop_dur=hop_dur)

        nb_blocks = 0
        last_block_nb_samples = None
        while True:
            data = reader.read()
            if data is not None:
                nb_blocks += 1
                last_block_nb_samples = len(data) // reader.sample_width
            else:
                break
        err_msg = "Wrong number of blocks read from source, expected: "
        err_msg += f"{expected_nb_blocks}, found: {nb_blocks}"
        assert nb_blocks == expected_nb_blocks, err_msg

        err_msg = (
            "Wrong number of samples in last block read from source, expected: "
        )
        err_msg += (
            f"{expected_last_block_nb_samples}, found: {last_block_nb_samples}"
        )

        assert last_block_nb_samples == expected_last_block_nb_samples, err_msg

    def test_hop_duration_exception(self):
        """Test passing hop_dur > block_dur raises ValueError"""
        with pytest.raises(ValueError):
            AudioReader(self.audio_source, block_dur=0.01, hop_dur=0.015)

    @pytest.mark.parametrize(
        "block_dur, hop_dur",
        [
            (None, None),  # default
            (0.01, None),  # block_dur_10ms_hop_dur_None
            (None, 0.01),  # block_dur_None__hop_dur_10ms
            (0.05, 0.05),  # block_dur_50ms_hop_dur_50ms
        ],
        ids=[
            "default",
            "block_dur_10ms_hop_dur_None",
            "block_dur_None__hop_dur_10ms",
            "block_dur_50ms_hop_dur_50ms",
        ],
    )
    def test_reader_class_block_dur_equals_hop_dur(self, block_dur, hop_dur):
        """Test passing hop_dur == block_dur does not create an instance of
        '_OverlapAudioReader'.
        """
        if block_dur is not None:
            reader = AudioReader(
                input=self.audio_source, block_dur=block_dur, hop_dur=hop_dur
            )
        else:
            reader = AudioReader(input=self.audio_source, hop_dur=hop_dur)
        assert not isinstance(reader, _OverlapAudioReader)

    def test_sampling_rate(self):
        reader = AudioReader(input=self.audio_source)
        sampling_rate = reader.sampling_rate
        assert (
            sampling_rate == 16000
        ), f"Wrong sampling rate, expected: 16000, found: {sampling_rate}"

    def test_sample_width(self):
        reader = AudioReader(input=self.audio_source)
        sample_width = reader.sample_width
        assert (
            sample_width == 2
        ), f"Wrong sample width, expected: 2, found: {sample_width}"

    def test_channels(self):
        reader = AudioReader(input=self.audio_source)
        channels = reader.channels
        assert (
            channels == 1
        ), f"Wrong number of channels, expected: 1, found: {channels}"

    def test_read(self):
        reader = AudioReader(input=self.audio_source, block_dur=0.02)
        reader_data = reader.read()
        audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        audio_source.open()
        audio_source_data = audio_source.read(320)
        audio_source.close()
        assert (
            reader_data == audio_source_data
        ), "Unexpected data read from AudioReader"

    def test_read_with_overlap(self):
        reader = AudioReader(
            input=self.audio_source, block_dur=0.02, hop_dur=0.01
        )
        _ = reader.read()  # first block
        reader_data = reader.read()  # second block with 0.01 S overlap
        audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        audio_source.open()
        _ = audio_source.read(160)
        audio_source_data = audio_source.read(320)
        audio_source.close()
        assert (
            reader_data == audio_source_data
        ), "Unexpected data read from AudioReader"

    def test_read_from_AudioReader_with_max_read(self):
        # read a maximum of 0.75 seconds from audio source
        reader = AudioReader(input=self.audio_source, max_read=0.75)
        assert isinstance(reader._audio_source._audio_source, _Limiter)
        reader_data = _read_all_data(reader)

        audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        audio_source.open()
        audio_source_data = audio_source.read(int(16000 * 0.75))
        audio_source.close()

        assert (
            reader_data == audio_source_data
        ), f"Unexpected data read from AudioReader with 'max_read = {0.75}'"

    def test_read_data_size_from_AudioReader_with_max_read(self):
        # read a maximum of 1.191 seconds from audio source
        reader = AudioReader(input=self.audio_source, max_read=1.191)
        assert isinstance(reader._audio_source._audio_source, _Limiter)
        total_samples = round(reader.sampling_rate * 1.191)
        block_size = int(reader.block_dur * reader.sampling_rate)
        nb_full_blocks, last_block_size = divmod(total_samples, block_size)
        total_samples_with_overlap = (
            nb_full_blocks * block_size + last_block_size
        )
        expected_read_bytes = (
            total_samples_with_overlap * reader.sample_width * reader.channels
        )

        reader_data = _read_all_data(reader)
        total_read = len(reader_data)
        err_msg = f"Wrong data length read from LimiterADS, expected: {expected_read_bytes}, found: {total_read}"
        assert total_read == expected_read_bytes, err_msg

    def test_read_from_Recorder(self):
        reader = Recorder(input=self.audio_source, block_dur=0.025)
        reader_data = []
        for _ in range(10):
            block = reader.read()
            if block is None:
                break
            reader_data.append(block)
        reader_data = b"".join(reader_data)

        audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        audio_source.open()
        audio_source_data = audio_source.read(400 * 10)
        audio_source.close()

        assert (
            reader_data == audio_source_data
        ), "Unexpected data read from Recorder"

    def test_AudioReader_rewindable(self):
        reader = AudioReader(input=self.audio_source, record=True)
        assert (
            reader.rewindable
        ), "AudioReader with record=True should be rewindable"

    def test_AudioReader_record_and_rewind(self):
        reader = AudioReader(
            input=self.audio_source, record=True, block_dur=0.02
        )
        # read 0.02 * 10 = 0.2 sec. of data
        for i in range(10):
            reader.read()
        reader.rewind()

        # read all available data after rewind
        reader_data = _read_all_data(reader)

        audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        audio_source.open()
        audio_source_data = audio_source.read(320 * 10)  # read 0.2 sec. of data
        audio_source.close()

        assert (
            reader_data == audio_source_data
        ), "Unexpected data read from AudioReader with record = True"

    def test_Recorder_record_and_rewind(self):
        recorder = Recorder(input=self.audio_source, block_dur=0.02)
        # read 0.02 * 10 = 0.2 sec. of data
        for i in range(10):
            recorder.read()

        recorder.rewind()

        # read all available data after rewind
        recorder_data = []
        recorder_data = _read_all_data(recorder)

        audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        audio_source.open()
        audio_source_data = audio_source.read(320 * 10)  # read 0.2 sec. of data
        audio_source.close()

        assert (
            recorder_data == audio_source_data
        ), "Unexpected data read from Recorder"

    def test_read_overlapping_blocks(self):
        # Use arbitrary valid block_size and hop_size
        block_size = 1714
        hop_size = 313
        block_dur = block_size / self.audio_source.sampling_rate
        hop_dur = hop_size / self.audio_source.sampling_rate

        reader = AudioReader(
            input=self.audio_source,
            block_dur=block_dur,
            hop_dur=hop_dur,
        )

        # Read all available overlapping blocks of data
        reader_data = []
        while True:
            block = reader.read()
            if block is None:
                break
            reader_data.append(block)

        # Read all data from file and build a BufferAudioSource
        fp = wave.open(dataset.one_to_six_arabic_16000_mono_bc_noise, "r")
        wave_data = fp.readframes(fp.getnframes())
        fp.close()
        audio_source = BufferAudioSource(
            wave_data,
            reader.sampling_rate,
            reader.sample_width,
            reader.channels,
        )
        audio_source.open()

        # Compare all blocks read from OverlapADS to those read from an
        # audio source with a manual position setting
        for i, block in enumerate(reader_data):
            tmp = audio_source.read(block_size)
            assert (
                block == tmp
            ), f"Unexpected data (block {i}) from reader with overlapping blocks"
            audio_source.position = (i + 1) * hop_size

        audio_source.close()

    def test_read_overlapping_blocks_with_max_read(self):
        block_size = 256
        hop_size = 200
        block_dur = block_size / self.audio_source.sampling_rate
        hop_dur = hop_size / self.audio_source.sampling_rate

        reader = AudioReader(
            input=self.audio_source,
            block_dur=block_dur,
            hop_dur=hop_dur,
            max_read=0.5,
        )

        # Read all available overlapping blocks of data
        reader_data = []
        while True:
            block = reader.read()
            if block is None:
                break
            reader_data.append(block)

        # Read all data from file and build a BufferAudioSource
        fp = wave.open(dataset.one_to_six_arabic_16000_mono_bc_noise, "r")
        wave_data = fp.readframes(fp.getnframes())
        fp.close()
        audio_source = BufferAudioSource(
            wave_data,
            reader.sampling_rate,
            reader.sample_width,
            reader.channels,
        )
        audio_source.open()

        # Compare all blocks read from OverlapADS to those read from an
        # audio source with a manual position setting
        for i, block in enumerate(reader_data):
            tmp = audio_source.read(len(block) // (reader.sw * reader.ch))
            assert (
                block == tmp
            ), f"Unexpected data (block {i}) from reader with overlapping blocks and max_read"
            audio_source.position = (i + 1) * hop_size

        audio_source.close()

    def test_length_read_overlapping_blocks_with_max_read(self):
        block_size = 313
        hop_size = 207
        block_dur = block_size / self.audio_source.sampling_rate
        hop_dur = hop_size / self.audio_source.sampling_rate

        reader = AudioReader(
            input=self.audio_source,
            max_read=1.932,
            block_dur=block_dur,
            hop_dur=hop_dur,
        )

        total_samples = round(reader.sampling_rate * 1.932)
        first_read_size = block_size
        next_read_size = block_size - hop_size
        nb_next_blocks, last_block_size = divmod(
            (total_samples - first_read_size), next_read_size
        )
        total_samples_with_overlap = (
            first_read_size + next_read_size * nb_next_blocks + last_block_size
        )
        expected_read_bytes = (
            total_samples_with_overlap * reader.sw * reader.channels
        )

        cache_size = (
            (block_size - hop_size) * reader.sample_width * reader.channels
        )
        total_read = cache_size

        i = 0
        while True:
            block = reader.read()
            if block is None:
                break
            i += 1
            total_read += len(block) - cache_size

        err_msg = (
            "Wrong data length read from LimiterADS, expected: {0}, found: {1}"
        )
        assert total_read == expected_read_bytes, err_msg.format(
            expected_read_bytes, total_read
        )

    def test_reader_with_overlapping_blocks__rewindable(self):
        reader = AudioReader(
            input=self.audio_source,
            block_dur=320,
            hop_dur=160,
            record=True,
        )
        assert (
            reader.rewindable
        ), "AudioReader with record=True should be rewindable"

    def test_overlapping_blocks_with_max_read_rewind_and_read(self):
        # Use arbitrary valid block_size and hop_size
        block_size = 1600
        hop_size = 400
        block_dur = block_size / self.audio_source.sampling_rate
        hop_dur = hop_size / self.audio_source.sampling_rate

        reader = AudioReader(
            input=self.audio_source,
            block_dur=block_dur,
            hop_dur=hop_dur,
            record=True,
        )

        # Read all available data overlapping blocks
        i = 0
        while True:
            block = reader.read()
            if block is None:
                break
            i += 1

        reader.rewind()

        # Read all data from file and build a BufferAudioSource
        fp = wave.open(dataset.one_to_six_arabic_16000_mono_bc_noise, "r")
        wave_data = fp.readframes(fp.getnframes())
        fp.close()
        audio_source = BufferAudioSource(
            wave_data,
            reader.sampling_rate,
            reader.sample_width,
            reader.channels,
        )
        audio_source.open()

        # Compare blocks read from AudioReader to those read from an BufferAudioSource with manual position setting
        for j in range(i):
            tmp = audio_source.read(block_size)
            assert (
                reader.read() == tmp
            ), f"Unexpected data (block {i}) from reader with overlapping blocks and record = True"
            audio_source.position = (j + 1) * hop_size

        audio_source.close()

    def test_overlapping_blocks_with_record_and_max_read_rewind_and_read(self):
        # Use arbitrary valid block_size and hop_size
        block_size = 1600
        hop_size = 400
        block_dur = block_size / self.audio_source.sampling_rate
        hop_dur = hop_size / self.audio_source.sampling_rate

        reader = AudioReader(
            input=self.audio_source,
            max_time=1.50,
            block_dur=block_dur,
            hop_dur=hop_dur,
            record=True,
        )

        # Read all available data overlapping blocks
        i = 0
        while True:
            block = reader.read()
            if block is None:
                break
            i += 1

        reader.rewind()

        # Read all data from file and build a BufferAudioSource
        fp = wave.open(dataset.one_to_six_arabic_16000_mono_bc_noise, "r")
        wave_data = fp.readframes(fp.getnframes())
        fp.close()
        audio_source = BufferAudioSource(
            wave_data,
            reader.sampling_rate,
            reader.sample_width,
            reader.channels,
        )
        audio_source.open()

        # Compare all blocks read from AudioReader to those read from BufferAudioSource with a manual position setting
        for j in range(i):
            tmp = audio_source.read(block_size)
            assert (
                reader.read() == tmp
            ), "Unexpected block (N={0}) read from OverlapADS".format(i)
            audio_source.position = (j + 1) * hop_size

        audio_source.close()

    def test_length_read_overlapping_blocks_with_record_and_max_read(self):
        # Use arbitrary valid block_size and hop_size
        block_size = 1000
        hop_size = 200
        block_dur = block_size / self.audio_source.sampling_rate
        hop_dur = hop_size / self.audio_source.sampling_rate

        reader = AudioReader(
            input=self.audio_source,
            block_dur=block_dur,
            hop_dur=hop_dur,
            record=True,
            max_read=1.317,
        )
        total_samples = round(reader.sampling_rate * 1.317)
        first_read_size = block_size
        next_read_size = block_size - hop_size
        nb_next_blocks, last_block_size = divmod(
            (total_samples - first_read_size), next_read_size
        )
        total_samples_with_overlap = (
            first_read_size + next_read_size * nb_next_blocks + last_block_size
        )
        expected_read_bytes = (
            total_samples_with_overlap * reader.sample_width * reader.channels
        )

        cache_size = (
            (block_size - hop_size) * reader.sample_width * reader.channels
        )
        total_read = cache_size

        i = 0
        while True:
            block = reader.read()
            if block is None:
                break
            i += 1
            total_read += len(block) - cache_size

        err_msg = f"Wrong data length read from AudioReader, expected: {expected_read_bytes}, found: {total_read}"
        assert total_read == expected_read_bytes, err_msg


def test_AudioReader_raw_data():

    data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    block_size = 5
    hop_size = 4
    reader = AudioReader(
        input=data,
        sampling_rate=16,
        sample_width=2,
        channels=1,
        block_dur=block_size / 16,
        hop_dur=hop_size / 16,
        max_read=0.80,
        record=True,
    )
    reader.open()

    assert (
        reader.sampling_rate == 16
    ), f"Wrong sampling rate, expected: 16, found: {reader.sampling_rate }"

    assert (
        reader.sample_width == 2
    ), f"Wrong sample width, expected: 2, found: {reader.sample_width}"

    # Read all available data overlapping blocks
    i = 0
    while True:
        block = reader.read()
        if block is None:
            break
        i += 1

    reader.rewind()

    # Build a BufferAudioSource
    audio_source = BufferAudioSource(
        data, reader.sampling_rate, reader.sample_width, reader.channels
    )
    audio_source.open()

    # Compare all blocks read from AudioReader to those read from an audio
    # source with a manual position setting
    for j in range(i):
        tmp = audio_source.read(block_size)
        block = reader.read()
        assert (
            block == tmp
        ), f"Unexpected block '{block}' (N={i}) read from OverlapADS"
        audio_source.position = (j + 1) * hop_size
    audio_source.close()
    reader.close()


def test_AudioReader_alias_params():
    reader = AudioReader(
        input=b"0" * 1600,
        sr=16000,
        sw=2,
        channels=1,
    )
    assert reader.sampling_rate == 16000, (
        "Unexpected sampling rate: reader.sampling_rate = "
        + f"{reader.sampling_rate} instead of 16000"
    )
    assert reader.sr == 16000, (
        "Unexpected sampling rate: reader.sr = "
        + f"{reader.sr} instead of 16000"
    )
    assert reader.sample_width == 2, (
        "Unexpected sample width: reader.sample_width = "
        + f"{reader.sample_width} instead of 2"
    )
    assert reader.sw == 2, (
        "Unexpected sample width: reader.sw = " + f"{reader.sw} instead of 2"
    )
    assert reader.channels == 1, (
        "Unexpected number of channels: reader.channels = "
        + f"{reader.channels} instead of 1"
    )
    assert reader.ch == 1, (
        "Unexpected number of channels: reader.ch = "
        + f"{reader.ch} instead of 1"
    )


@pytest.mark.parametrize(
    "file_id, max_read, size",
    [
        ("mono_400", 0.5, 16000),  # mono
        ("3channel_400-800-1600", 0.5, 16000 * 3),  # multichannel
    ],
    ids=["mono", "multichannel"],
)
def test_Limiter(file_id, max_read, size):
    input_wav = "tests/data/test_16KHZ_{}Hz.wav".format(file_id)
    input_raw = "tests/data/test_16KHZ_{}Hz.raw".format(file_id)
    with open(input_raw, "rb") as fp:
        expected = fp.read(size)

    reader = AudioReader(input_wav, block_dur=0.1, max_read=max_read)
    reader.open()
    data = _read_all_data(reader)
    reader.close()
    assert data == expected


@pytest.mark.parametrize(
    "file_id",
    [
        "mono_400",  # mono
        "3channel_400-800-1600",  # multichannel
    ],
    ids=["mono", "multichannel"],
)
def test_Recorder(file_id):
    input_wav = "tests/data/test_16KHZ_{}Hz.wav".format(file_id)
    input_raw = "tests/data/test_16KHZ_{}Hz.raw".format(file_id)
    with open(input_raw, "rb") as fp:
        expected = fp.read()

    reader = AudioReader(input_wav, block_dur=0.1, record=True)
    reader.open()
    data = _read_all_data(reader)
    assert data == expected

    # rewind many times
    for _ in range(3):
        reader.rewind()
        data = _read_all_data(reader)
        assert data == expected
        assert data == reader.data
    reader.close()


@pytest.mark.parametrize(
    "file_id",
    [
        "mono_400",  # mono
        "3channel_400-800-1600",  # multichannel
    ],
    ids=["mono", "multichannel"],
)
def test_Recorder_alias(file_id):
    input_wav = "tests/data/test_16KHZ_{}Hz.wav".format(file_id)
    input_raw = "tests/data/test_16KHZ_{}Hz.raw".format(file_id)
    with open(input_raw, "rb") as fp:
        expected = fp.read()

    reader = Recorder(input_wav, block_dur=0.1)
    reader.open()
    data = _read_all_data(reader)
    assert data == expected

    # rewind many times
    for _ in range(3):
        reader.rewind()
        data = _read_all_data(reader)
        assert data == expected
        assert data == reader.data
    reader.close()
