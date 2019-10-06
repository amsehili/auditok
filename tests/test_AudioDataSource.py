"""
@author: Amine Sehili <amine.sehili@gmail.com>
September 2015

"""

import unittest
from functools import partial
import sys
import wave
from genty import genty, genty_dataset
from auditok import (
    dataset,
    ADSFactory,
    AudioDataSource,
    BufferAudioSource,
    WaveAudioSource,
    DuplicateArgument,
)


class TestADSFactoryFileAudioSource(unittest.TestCase):
    def setUp(self):
        self.audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )

    def test_ADS_type(self):

        ads = ADSFactory.ads(audio_source=self.audio_source)

        self.assertIsInstance(
            ads,
            AudioDataSource,
            msg="wrong type for ads object, expected: 'AudioDataSource', found: {0}".format(
                type(ads)
            ),
        )

    def test_default_block_size(self):
        ads = ADSFactory.ads(audio_source=self.audio_source)
        size = ads.block_size
        self.assertEqual(
            size,
            160,
            "Wrong default block_size, expected: 160, found: {0}".format(size),
        )

    def test_block_size(self):
        ads = ADSFactory.ads(audio_source=self.audio_source, block_size=512)
        size = ads.block_size
        self.assertEqual(
            size,
            512,
            "Wrong block_size, expected: 512, found: {0}".format(size),
        )

        # with alias keyword
        ads = ADSFactory.ads(audio_source=self.audio_source, bs=160)
        size = ads.block_size
        self.assertEqual(
            size,
            160,
            "Wrong block_size, expected: 160, found: {0}".format(size),
        )

    def test_block_duration(self):

        ads = ADSFactory.ads(
            audio_source=self.audio_source, block_dur=0.01
        )  # 10 ms
        size = ads.block_size
        self.assertEqual(
            size,
            160,
            "Wrong block_size, expected: 160, found: {0}".format(size),
        )

        # with alias keyword
        ads = ADSFactory.ads(audio_source=self.audio_source, bd=0.025)  # 25 ms
        size = ads.block_size
        self.assertEqual(
            size,
            400,
            "Wrong block_size, expected: 400, found: {0}".format(size),
        )

    def test_hop_duration(self):

        ads = ADSFactory.ads(
            audio_source=self.audio_source, block_dur=0.02, hop_dur=0.01
        )  # 10 ms
        size = ads.hop_size
        self.assertEqual(
            size, 160, "Wrong hop_size, expected: 160, found: {0}".format(size)
        )

        # with alias keyword
        ads = ADSFactory.ads(
            audio_source=self.audio_source, bd=0.025, hop_dur=0.015
        )  # 15 ms
        size = ads.hop_size
        self.assertEqual(
            size,
            240,
            "Wrong block_size, expected: 240, found: {0}".format(size),
        )

    def test_sampling_rate(self):
        ads = ADSFactory.ads(audio_source=self.audio_source)

        srate = ads.sampling_rate
        self.assertEqual(
            srate,
            16000,
            "Wrong sampling rate, expected: 16000, found: {0}".format(srate),
        )

    def test_sample_width(self):
        ads = ADSFactory.ads(audio_source=self.audio_source)

        swidth = ads.sample_width
        self.assertEqual(
            swidth,
            2,
            "Wrong sample width, expected: 2, found: {0}".format(swidth),
        )

    def test_channels(self):
        ads = ADSFactory.ads(audio_source=self.audio_source)

        channels = ads.channels
        self.assertEqual(
            channels,
            1,
            "Wrong number of channels, expected: 1, found: {0}".format(
                channels
            ),
        )

    def test_read(self):
        ads = ADSFactory.ads(audio_source=self.audio_source, block_size=256)

        ads.open()
        ads_data = ads.read()
        ads.close()

        audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        audio_source.open()
        audio_source_data = audio_source.read(256)
        audio_source.close()

        self.assertEqual(
            ads_data, audio_source_data, "Unexpected data read from ads"
        )

    def test_Limiter_Deco_read(self):
        # read a maximum of 0.75 seconds from audio source
        ads = ADSFactory.ads(audio_source=self.audio_source, max_time=0.75)

        ads_data = []
        ads.open()
        while True:
            block = ads.read()
            if block is None:
                break
            ads_data.append(block)
        ads.close()
        ads_data = b"".join(ads_data)

        audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        audio_source.open()
        audio_source_data = audio_source.read(int(16000 * 0.75))
        audio_source.close()

        self.assertEqual(
            ads_data, audio_source_data, "Unexpected data read from LimiterADS"
        )

    def test_Limiter_Deco_read_limit(self):
        # read a maximum of 1.191 seconds from audio source
        ads = ADSFactory.ads(audio_source=self.audio_source, max_time=1.191)
        total_samples = round(ads.sampling_rate * 1.191)
        nb_full_blocks, last_block_size = divmod(total_samples, ads.block_size)
        total_samples_with_overlap = (
            nb_full_blocks * ads.block_size + last_block_size
        )
        expected_read_bytes = (
            total_samples_with_overlap * ads.sw * ads.channels
        )

        total_read = 0
        ads.open()
        i = 0
        while True:
            block = ads.read()
            if block is None:
                break
            i += 1
            total_read += len(block)

        ads.close()

        self.assertEqual(
            total_read,
            expected_read_bytes,
            "Wrong data length read from LimiterADS, expected: {0}, found: {1}".format(
                expected_read_bytes, total_read
            ),
        )

    def test_Recorder_Deco_read(self):
        ads = ADSFactory.ads(
            audio_source=self.audio_source, record=True, block_size=500
        )

        ads_data = []
        ads.open()
        for i in range(10):
            block = ads.read()
            if block is None:
                break
            ads_data.append(block)
        ads.close()
        ads_data = b"".join(ads_data)

        audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        audio_source.open()
        audio_source_data = audio_source.read(500 * 10)
        audio_source.close()

        self.assertEqual(
            ads_data,
            audio_source_data,
            "Unexpected data read from RecorderADS",
        )

    def test_Recorder_Deco_is_rewindable(self):
        ads = ADSFactory.ads(audio_source=self.audio_source, record=True)

        self.assertTrue(
            ads.rewindable, "RecorderADS.is_rewindable should return True"
        )

    def test_Recorder_Deco_rewind_and_read(self):
        ads = ADSFactory.ads(
            audio_source=self.audio_source, record=True, block_size=320
        )

        ads.open()
        for i in range(10):
            ads.read()

        ads.rewind()

        # read all available data after rewind
        ads_data = []
        while True:
            block = ads.read()
            if block is None:
                break
            ads_data.append(block)
        ads.close()
        ads_data = b"".join(ads_data)

        audio_source = WaveAudioSource(
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise
        )
        audio_source.open()
        audio_source_data = audio_source.read(320 * 10)
        audio_source.close()

        self.assertEqual(
            ads_data,
            audio_source_data,
            "Unexpected data read from RecorderADS",
        )

    def test_Overlap_Deco_read(self):

        # Use arbitrary valid block_size and hop_size
        block_size = 1714
        hop_size = 313

        ads = ADSFactory.ads(
            audio_source=self.audio_source,
            block_size=block_size,
            hop_size=hop_size,
        )

        # Read all available data overlapping blocks
        ads.open()
        ads_data = []
        while True:
            block = ads.read()
            if block is None:
                break
            ads_data.append(block)
        ads.close()

        # Read all data from file and build a BufferAudioSource
        fp = wave.open(dataset.one_to_six_arabic_16000_mono_bc_noise, "r")
        wave_data = fp.readframes(fp.getnframes())
        fp.close()
        audio_source = BufferAudioSource(
            wave_data, ads.sampling_rate, ads.sample_width, ads.channels
        )
        audio_source.open()

        # Compare all blocks read from OverlapADS to those read
        # from an audio source with a manual position setting
        for i, block in enumerate(ads_data):

            tmp = audio_source.read(block_size)

            self.assertEqual(
                block,
                tmp,
                "Unexpected block (N={0}) read from OverlapADS".format(i),
            )

            audio_source.position = (i + 1) * hop_size

        audio_source.close()

    def test_Limiter_Overlap_Deco_read(self):

        block_size = 256
        hop_size = 200

        ads = ADSFactory.ads(
            audio_source=self.audio_source,
            max_time=0.50,
            block_size=block_size,
            hop_size=hop_size,
        )

        # Read all available data overlapping blocks
        ads.open()
        ads_data = []
        while True:
            block = ads.read()
            if block is None:
                break
            ads_data.append(block)
        ads.close()

        # Read all data from file and build a BufferAudioSource
        fp = wave.open(dataset.one_to_six_arabic_16000_mono_bc_noise, "r")
        wave_data = fp.readframes(fp.getnframes())
        fp.close()
        audio_source = BufferAudioSource(
            wave_data, ads.sampling_rate, ads.sample_width, ads.channels
        )
        audio_source.open()

        # Compare all blocks read from OverlapADS to those read
        # from an audio source with a manual position setting
        for i, block in enumerate(ads_data):
            tmp = audio_source.read(len(block) // (ads.sw * ads.ch))
            self.assertEqual(
                len(block),
                len(tmp),
                "Unexpected block (N={0}) read from OverlapADS".format(i),
            )
            audio_source.position = (i + 1) * hop_size

        audio_source.close()

    def test_Limiter_Overlap_Deco_read_limit(self):

        block_size = 313
        hop_size = 207
        ads = ADSFactory.ads(
            audio_source=self.audio_source,
            max_time=1.932,
            block_size=block_size,
            hop_size=hop_size,
        )

        total_samples = round(ads.sampling_rate * 1.932)
        first_read_size = block_size
        next_read_size = block_size - hop_size
        nb_next_blocks, last_block_size = divmod(
            (total_samples - first_read_size), next_read_size
        )
        total_samples_with_overlap = (
            first_read_size + next_read_size * nb_next_blocks + last_block_size
        )
        expected_read_bytes = (
            total_samples_with_overlap * ads.sw * ads.channels
        )

        cache_size = (block_size - hop_size) * ads.sample_width * ads.channels
        total_read = cache_size

        ads.open()
        i = 0
        while True:
            block = ads.read()
            if block is None:
                break
            i += 1
            total_read += len(block) - cache_size

        ads.close()
        self.assertEqual(
            total_read,
            expected_read_bytes,
            "Wrong data length read from LimiterADS, expected: {0}, found: {1}".format(
                expected_read_bytes, total_read
            ),
        )

    def test_Recorder_Overlap_Deco_is_rewindable(self):
        ads = ADSFactory.ads(
            audio_source=self.audio_source,
            block_size=320,
            hop_size=160,
            record=True,
        )
        self.assertTrue(
            ads.rewindable, "RecorderADS.is_rewindable should return True"
        )

    def test_Recorder_Overlap_Deco_rewind_and_read(self):

        # Use arbitrary valid block_size and hop_size
        block_size = 1600
        hop_size = 400

        ads = ADSFactory.ads(
            audio_source=self.audio_source,
            block_size=block_size,
            hop_size=hop_size,
            record=True,
        )

        # Read all available data overlapping blocks
        ads.open()
        i = 0
        while True:
            block = ads.read()
            if block is None:
                break
            i += 1

        ads.rewind()

        # Read all data from file and build a BufferAudioSource
        fp = wave.open(dataset.one_to_six_arabic_16000_mono_bc_noise, "r")
        wave_data = fp.readframes(fp.getnframes())
        fp.close()
        audio_source = BufferAudioSource(
            wave_data, ads.sampling_rate, ads.sample_width, ads.channels
        )
        audio_source.open()

        # Compare all blocks read from OverlapADS to those read
        # from an audio source with a manual position setting
        for j in range(i):

            tmp = audio_source.read(block_size)

            self.assertEqual(
                ads.read(),
                tmp,
                "Unexpected block (N={0}) read from OverlapADS".format(i),
            )
            audio_source.position = (j + 1) * hop_size

        ads.close()
        audio_source.close()

    def test_Limiter_Recorder_Overlap_Deco_rewind_and_read(self):

        # Use arbitrary valid block_size and hop_size
        block_size = 1600
        hop_size = 400

        ads = ADSFactory.ads(
            audio_source=self.audio_source,
            max_time=1.50,
            block_size=block_size,
            hop_size=hop_size,
            record=True,
        )

        # Read all available data overlapping blocks
        ads.open()
        i = 0
        while True:
            block = ads.read()
            if block is None:
                break
            i += 1

        ads.rewind()

        # Read all data from file and build a BufferAudioSource
        fp = wave.open(dataset.one_to_six_arabic_16000_mono_bc_noise, "r")
        wave_data = fp.readframes(fp.getnframes())
        fp.close()
        audio_source = BufferAudioSource(
            wave_data, ads.sampling_rate, ads.sample_width, ads.channels
        )
        audio_source.open()

        # Compare all blocks read from OverlapADS to those read
        # from an audio source with a manual position setting
        for j in range(i):

            tmp = audio_source.read(block_size)

            self.assertEqual(
                ads.read(),
                tmp,
                "Unexpected block (N={0}) read from OverlapADS".format(i),
            )
            audio_source.position = (j + 1) * hop_size

        ads.close()
        audio_source.close()

    def test_Limiter_Recorder_Overlap_Deco_rewind_and_read_limit(self):

        # Use arbitrary valid block_size and hop_size
        block_size = 1000
        hop_size = 200

        ads = ADSFactory.ads(
            audio_source=self.audio_source,
            max_time=1.317,
            block_size=block_size,
            hop_size=hop_size,
            record=True,
        )
        total_samples = round(ads.sampling_rate * 1.317)
        first_read_size = block_size
        next_read_size = block_size - hop_size
        nb_next_blocks, last_block_size = divmod(
            (total_samples - first_read_size), next_read_size
        )
        total_samples_with_overlap = (
            first_read_size + next_read_size * nb_next_blocks + last_block_size
        )
        expected_read_bytes = (
            total_samples_with_overlap * ads.sw * ads.channels
        )

        cache_size = (block_size - hop_size) * ads.sample_width * ads.channels
        total_read = cache_size

        ads.open()
        i = 0
        while True:
            block = ads.read()
            if block is None:
                break
            i += 1
            total_read += len(block) - cache_size

        ads.close()
        self.assertEqual(
            total_read,
            expected_read_bytes,
            "Wrong data length read from LimiterADS, expected: {0}, found: {1}".format(
                expected_read_bytes, total_read
            ),
        )


class TestADSFactoryBufferAudioSource(unittest.TestCase):
    def setUp(self):
        self.signal = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        self.ads = ADSFactory.ads(
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            block_size=4,
        )

    def test_ADS_BAS_sampling_rate(self):
        srate = self.ads.sampling_rate
        self.assertEqual(
            srate,
            16,
            "Wrong sampling rate, expected: 16000, found: {0}".format(srate),
        )

    def test_ADS_BAS_get_sample_width(self):
        swidth = self.ads.sample_width
        self.assertEqual(
            swidth,
            2,
            "Wrong sample width, expected: 2, found: {0}".format(swidth),
        )

    def test_ADS_BAS_get_channels(self):
        channels = self.ads.channels
        self.assertEqual(
            channels,
            1,
            "Wrong number of channels, expected: 1, found: {0}".format(
                channels
            ),
        )

    def test_Limiter_Recorder_Overlap_Deco_rewind_and_read(self):

        # Use arbitrary valid block_size and hop_size
        block_size = 5
        hop_size = 4

        ads = ADSFactory.ads(
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            max_time=0.80,
            block_size=block_size,
            hop_size=hop_size,
            record=True,
        )

        # Read all available data overlapping blocks
        ads.open()
        i = 0
        while True:
            block = ads.read()
            if block is None:
                break
            i += 1

        ads.rewind()

        # Build a BufferAudioSource
        audio_source = BufferAudioSource(
            self.signal, ads.sampling_rate, ads.sample_width, ads.channels
        )
        audio_source.open()

        # Compare all blocks read from OverlapADS to those read
        # from an audio source with a manual position setting
        for j in range(i):

            tmp = audio_source.read(block_size)

            block = ads.read()

            self.assertEqual(
                block,
                tmp,
                "Unexpected block '{}' (N={}) read from OverlapADS".format(
                    block, i
                ),
            )
            audio_source.position = (j + 1) * hop_size

        ads.close()
        audio_source.close()


class TestADSFactoryAlias(unittest.TestCase):
    def setUp(self):
        self.signal = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"

    def test_sampling_rate_alias(self):
        ads = ADSFactory.ads(
            data_buffer=self.signal,
            sr=16,
            sample_width=2,
            channels=1,
            block_dur=0.5,
        )
        srate = ads.sampling_rate
        self.assertEqual(
            srate,
            16,
            "Wrong sampling rate, expected: 16000, found: {0}".format(srate),
        )

    def test_sampling_rate_duplicate(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sr=16,
            sampling_rate=16,
            sample_width=2,
            channels=1,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_sample_width_alias(self):
        ads = ADSFactory.ads(
            data_buffer=self.signal,
            sampling_rate=16,
            sw=2,
            channels=1,
            block_dur=0.5,
        )
        swidth = ads.sample_width
        self.assertEqual(
            swidth,
            2,
            "Wrong sample width, expected: 2, found: {0}".format(swidth),
        )

    def test_sample_width_duplicate(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sampling_rate=16,
            sw=2,
            sample_width=2,
            channels=1,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_channels_alias(self):
        ads = ADSFactory.ads(
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            ch=1,
            block_dur=4,
        )
        channels = ads.channels
        self.assertEqual(
            channels,
            1,
            "Wrong number of channels, expected: 1, found: {0}".format(
                channels
            ),
        )

    def test_channels_duplicate(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            ch=1,
            channels=1,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_block_size_alias(self):
        ads = ADSFactory.ads(
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            bs=8,
        )
        size = ads.block_size
        self.assertEqual(
            size,
            8,
            "Wrong block_size using bs alias, expected: 8, found: {0}".format(
                size
            ),
        )

    def test_block_size_duplicate(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            bs=4,
            block_size=4,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_block_duration_alias(self):
        ads = ADSFactory.ads(
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            bd=0.75,
        )
        # 0.75 ms = 0.75 * 16 = 12
        size = ads.block_size
        self.assertEqual(
            size,
            12,
            "Wrong block_size set with a block_dur alias 'bd', expected: 8, found: {0}".format(
                size
            ),
        )

    def test_block_duration_duplicate(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            bd=4,
            block_dur=4,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_block_size_duration_duplicate(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            bd=4,
            bs=12,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_hop_duration_alias(self):

        ads = ADSFactory.ads(
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            bd=0.75,
            hd=0.5,
        )
        size = ads.hop_size
        self.assertEqual(
            size,
            8,
            "Wrong block_size using bs alias, expected: 8, found: {0}".format(
                size
            ),
        )

    def test_hop_duration_duplicate(self):

        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            bd=0.75,
            hd=0.5,
            hop_dur=0.5,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_hop_size_duration_duplicate(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            bs=8,
            hs=4,
            hd=1,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_hop_size_greater_than_block_size(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            bs=4,
            hs=8,
        )
        self.assertRaises(ValueError, func)

    def test_filename_alias(self):
        ads = ADSFactory.ads(fn=dataset.one_to_six_arabic_16000_mono_bc_noise)

    def test_filename_duplicate(self):

        func = partial(
            ADSFactory.ads,
            fn=dataset.one_to_six_arabic_16000_mono_bc_noise,
            filename=dataset.one_to_six_arabic_16000_mono_bc_noise,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_data_buffer_duplicate(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            db=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_max_time_alias(self):
        ads = ADSFactory.ads(
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            mt=10,
            block_dur=0.5,
        )
        self.assertEqual(
            ads.max_read,
            10,
            "Wrong AudioDataSource.max_read, expected: 10, found: {}".format(
                ads.max_read
            ),
        )

    def test_max_time_duplicate(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            mt=True,
            max_time=True,
        )

        self.assertRaises(DuplicateArgument, func)

    def test_record_alias(self):
        ads = ADSFactory.ads(
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            rec=True,
            block_dur=0.5,
        )
        self.assertTrue(
            ads.rewindable, "AudioDataSource.rewindable expected to be True"
        )

    def test_record_duplicate(self):
        func = partial(
            ADSFactory.ads,
            data_buffer=self.signal,
            sampling_rate=16,
            sample_width=2,
            channels=1,
            rec=True,
            record=True,
        )
        self.assertRaises(DuplicateArgument, func)

    def test_Limiter_Recorder_Overlap_Deco_rewind_and_read_alias(self):

        # Use arbitrary valid block_size and hop_size
        block_size = 5
        hop_size = 4

        ads = ADSFactory.ads(
            db=self.signal,
            sr=16,
            sw=2,
            ch=1,
            mt=0.80,
            bs=block_size,
            hs=hop_size,
            rec=True,
        )

        # Read all available data overlapping blocks
        ads.open()
        i = 0
        while True:
            block = ads.read()
            if block is None:
                break
            i += 1

        ads.rewind()

        # Build a BufferAudioSource
        audio_source = BufferAudioSource(
            self.signal, ads.sampling_rate, ads.sample_width, ads.channels
        )
        audio_source.open()

        # Compare all blocks read from AudioDataSource to those read
        # from an audio source with manual position definition
        for j in range(i):
            tmp = audio_source.read(block_size)
            block = ads.read()
            self.assertEqual(
                block,
                tmp,
                "Unexpected block (N={0}) read from OverlapADS".format(i),
            )
            audio_source.position = (j + 1) * hop_size
        ads.close()
        audio_source.close()


def _read_all_data(reader):
    blocks = []
    while True:
        data = reader.read()
        if data is None:
            break
        blocks.append(data)
    return b"".join(blocks)


@genty
class TestAudioReader(unittest.TestCase):

    # TODO move all tests here when backward compatibility
    # with ADSFactory is dropped

    @genty_dataset(
        mono=("mono_400", 0.5, 16000),
        multichannel=("3channel_400-800-1600", 0.5, 16000 * 3),
    )
    def test_Limiter(self, file_id, max_read, size):
        input_wav = "tests/data/test_16KHZ_{}Hz.wav".format(file_id)
        input_raw = "tests/data/test_16KHZ_{}Hz.raw".format(file_id)
        with open(input_raw, "rb") as fp:
            expected = fp.read(size)

        reader = AudioDataSource(input_wav, block_dur=0.1, max_read=max_read)
        reader.open()
        data = _read_all_data(reader)
        reader.close()
        self.assertEqual(data, expected)

    @genty_dataset(mono=("mono_400",), multichannel=("3channel_400-800-1600",))
    def test_Recorder(self, file_id):
        input_wav = "tests/data/test_16KHZ_{}Hz.wav".format(file_id)
        input_raw = "tests/data/test_16KHZ_{}Hz.raw".format(file_id)
        with open(input_raw, "rb") as fp:
            expected = fp.read()

        reader = AudioDataSource(input_wav, block_dur=0.1, record=True)
        reader.open()
        data = _read_all_data(reader)
        self.assertEqual(data, expected)

        # rewind many times
        for _ in range(3):
            reader.rewind()
            data = _read_all_data(reader)
            self.assertEqual(data, expected)
            self.assertEqual(data, reader.data)
        reader.close()


if __name__ == "__main__":
    unittest.main()
