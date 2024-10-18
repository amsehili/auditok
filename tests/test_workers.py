import os
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

import pytest

import auditok.workers
from auditok import AudioReader, AudioRegion, split, split_and_join_with_silence
from auditok.cmdline_util import make_logger
from auditok.workers import (
    AudioEventsJoinerWorker,
    CommandLineWorker,
    PlayerWorker,
    PrintWorker,
    RegionSaverWorker,
    StreamSaverWorker,
    TokenizerWorker,
)


@pytest.fixture
def audio_data_source():
    reader = AudioReader(
        input="tests/data/test_split_10HZ_mono.raw",
        block_dur=0.1,
        sr=10,
        sw=2,
        ch=1,
    )
    yield reader
    reader.close()


@pytest.fixture
def expected_detections():
    return [
        (0.2, 1.6),
        (1.7, 3.1),
        (3.4, 5.4),
        (5.4, 7.4),
        (7.4, 7.6),
    ]


def test_TokenizerWorker(audio_data_source, expected_detections):
    with TemporaryDirectory() as tmpdir:
        file = os.path.join(tmpdir, "file.log")
        logger = make_logger(file=file, name="test_TokenizerWorker")
        tokenizer = TokenizerWorker(
            audio_data_source,
            logger=logger,
            min_dur=0.3,
            max_dur=2,
            max_silence=0.2,
            drop_trailing_silence=False,
            strict_min_dur=False,
            eth=50,
        )
        tokenizer.start_all()
        tokenizer.join()
        with open(file) as fp:
            log_lines = fp.readlines()

    log_fmt = (
        "[DET]: Detection {} (start: {:.3f}, end: {:.3f}, duration: {:.3f})"
    )
    assert len(tokenizer.detections) == len(expected_detections)
    for i, (det, exp, log_line) in enumerate(
        zip(tokenizer.detections, expected_detections, log_lines, strict=True),
        1,
    ):
        start, end = exp
        exp_log_line = log_fmt.format(i, start, end, end - start)
        assert pytest.approx(det.start) == start
        assert pytest.approx(det.end) == end
        assert log_line[28:].strip() == exp_log_line


def test_PlayerWorker(audio_data_source, expected_detections):
    with TemporaryDirectory() as tmpdir:
        file = os.path.join(tmpdir, "file.log")
        logger = make_logger(file=file, name="test_RegionSaverWorker")
        player_mock = Mock()
        observers = [PlayerWorker(player_mock, logger=logger)]
        tokenizer = TokenizerWorker(
            audio_data_source,
            logger=logger,
            observers=observers,
            min_dur=0.3,
            max_dur=2,
            max_silence=0.2,
            drop_trailing_silence=False,
            strict_min_dur=False,
            eth=50,
        )
        tokenizer.start_all()
        tokenizer.join()
        tokenizer._observers[0].join()
        with open(file) as fp:
            log_lines = [
                line for line in fp.readlines() if line.startswith("[PLAY]")
            ]

    assert player_mock.play.called
    assert len(tokenizer.detections) == len(expected_detections)
    log_fmt = "[PLAY]: Detection {id} played"
    for i, (det, exp, log_line) in enumerate(
        zip(tokenizer.detections, expected_detections, log_lines, strict=False),
        1,
    ):
        start, end = exp
        exp_log_line = log_fmt.format(id=i)
        assert pytest.approx(det.start) == start
        assert pytest.approx(det.end) == end
        assert log_line[28:].strip() == exp_log_line


def test_RegionSaverWorker(audio_data_source, expected_detections):
    filename_format = "Region_{id}_{start:.6f}-{end:.3f}_{duration:.3f}.wav"
    with TemporaryDirectory() as tmpdir:
        file = os.path.join(tmpdir, "file.log")
        logger = make_logger(file=file, name="test_RegionSaverWorker")
        observers = [RegionSaverWorker(filename_format, logger=logger)]
        tokenizer = TokenizerWorker(
            audio_data_source,
            logger=logger,
            observers=observers,
            min_dur=0.3,
            max_dur=2,
            max_silence=0.2,
            drop_trailing_silence=False,
            strict_min_dur=False,
            eth=50,
        )
        with patch("auditok.core.AudioRegion.save") as patched_save:
            tokenizer.start_all()
            tokenizer.join()
            tokenizer._observers[0].join()
        with open(file) as fp:
            log_lines = [
                line for line in fp.readlines() if line.startswith("[SAVE]")
            ]

    expected_save_calls = [
        call(
            filename_format.format(
                id=i, start=exp[0], end=exp[1], duration=exp[1] - exp[0]
            ),
            None,
        )
        for i, exp in enumerate(expected_detections, 1)
    ]

    mock_calls = [
        c for i, c in enumerate(patched_save.mock_calls) if i % 2 == 0
    ]
    assert mock_calls == expected_save_calls
    assert len(tokenizer.detections) == len(expected_detections)

    log_fmt = "[SAVE]: Detection {id} saved as '{filename}'"
    for i, (det, exp, log_line) in enumerate(
        zip(tokenizer.detections, expected_detections, log_lines, strict=False),
        1,
    ):
        start, end = exp
        expected_filename = filename_format.format(
            id=i, start=start, end=end, duration=end - start
        )
        exp_log_line = log_fmt.format(id=i, filename=expected_filename)
        assert pytest.approx(det.start) == start
        assert pytest.approx(det.end) == end
        assert log_line[28:].strip() == exp_log_line


def test_CommandLineWorker(audio_data_source, expected_detections):
    command_format = "do nothing with"
    with TemporaryDirectory() as tmpdir:
        file = os.path.join(tmpdir, "file.log")
        logger = make_logger(file=file, name="test_CommandLineWorker")
        observers = [CommandLineWorker(command_format, logger=logger)]
        tokenizer = TokenizerWorker(
            audio_data_source,
            logger=logger,
            observers=observers,
            min_dur=0.3,
            max_dur=2,
            max_silence=0.2,
            drop_trailing_silence=False,
            strict_min_dur=False,
            eth=50,
        )
        with patch("auditok.workers.os.system") as patched_os_system:
            tokenizer.start_all()
            tokenizer.join()
            tokenizer._observers[0].join()
        with open(file) as fp:
            log_lines = [
                line for line in fp.readlines() if line.startswith("[COMMAND]")
            ]

    expected_save_calls = [call(command_format) for _ in expected_detections]
    assert patched_os_system.mock_calls == expected_save_calls
    assert len(tokenizer.detections) == len(expected_detections)
    log_fmt = "[COMMAND]: Detection {id} command '{command}'"
    for i, (det, exp, log_line) in enumerate(
        zip(tokenizer.detections, expected_detections, log_lines, strict=False),
        1,
    ):
        start, end = exp
        exp_log_line = log_fmt.format(id=i, command=command_format)
        assert pytest.approx(det.start) == start
        assert pytest.approx(det.end) == end
        assert log_line[28:].strip() == exp_log_line


def test_PrintWorker(audio_data_source, expected_detections):
    observers = [
        PrintWorker(print_format="[{id}] {start} {end}, dur: {duration}")
    ]
    tokenizer = TokenizerWorker(
        audio_data_source,
        observers=observers,
        min_dur=0.3,
        max_dur=2,
        max_silence=0.2,
        drop_trailing_silence=False,
        strict_min_dur=False,
        eth=50,
    )
    with patch("builtins.print") as patched_print:
        tokenizer.start_all()
        tokenizer.join()
        tokenizer._observers[0].join()

    expected_print_calls = [
        call(
            "[{}] {:.3f} {:.3f}, dur: {:.3f}".format(
                i, exp[0], exp[1], exp[1] - exp[0]
            )
        )
        for i, exp in enumerate(expected_detections, 1)
    ]
    assert patched_print.mock_calls == expected_print_calls
    assert len(tokenizer.detections) == len(expected_detections)
    for det, exp in zip(tokenizer.detections, expected_detections, strict=True):
        start, end = exp
        assert pytest.approx(det.start) == start
        assert pytest.approx(det.end) == end


def test_StreamSaverWorker_wav(audio_data_source):
    with TemporaryDirectory() as tmpdir:
        expected_filename = os.path.join(tmpdir, "output.wav")
        saver = StreamSaverWorker(audio_data_source, expected_filename)
        saver.start()

        tokenizer = TokenizerWorker(saver)
        tokenizer.start_all()
        tokenizer.join()
        saver.join()

        output_filename = saver.export_audio()
        region = AudioRegion.load(
            "tests/data/test_split_10HZ_mono.raw", sr=10, sw=2, ch=1
        )

        expected_region = AudioRegion.load(output_filename)
        assert output_filename == expected_filename
        assert region == expected_region
        assert saver.data == bytes(expected_region)


@pytest.mark.parametrize(
    "export_format",
    [
        "raw",  # raw
        "wav",  # wav
    ],
    ids=[
        "raw",
        "raw",
    ],
)
def test_StreamSaverWorker(audio_data_source, export_format):
    with TemporaryDirectory() as tmpdir:
        expected_filename = os.path.join(tmpdir, f"output.{export_format}")
        saver = StreamSaverWorker(
            audio_data_source, expected_filename, export_format=export_format
        )
        saver.start()
        tokenizer = TokenizerWorker(saver)
        tokenizer.start_all()
        tokenizer.join()
        saver.join()
        output_filename = saver.export_audio()
        region = AudioRegion.load(
            "tests/data/test_split_10HZ_mono.raw", sr=10, sw=2, ch=1
        )
        expected_region = AudioRegion.load(
            output_filename, sr=10, sw=2, ch=1, audio_format=export_format
        )
        assert output_filename == expected_filename
        assert region == expected_region
        assert saver.data == bytes(expected_region)


def test_StreamSaverWorker_encode_audio(audio_data_source):
    with TemporaryDirectory() as tmpdir:
        with patch("auditok.workers._run_subprocess") as patch_rsp:
            patch_rsp.return_value = (1, None, None)
            expected_filename = os.path.join(tmpdir, "output.ogg")
            tmp_expected_filename = expected_filename + ".wav"
            saver = StreamSaverWorker(audio_data_source, expected_filename)
            saver.start()
            tokenizer = TokenizerWorker(saver)
            tokenizer.start_all()
            tokenizer.join()
            saver.join()

            with pytest.raises(auditok.workers.AudioEncodingError) as ae_error:
                saver._encode_export_audio()

        warn_msg = "Couldn't save audio data in the desired format "
        warn_msg += "'ogg'.\nEither none of 'ffmpeg', 'avconv' or 'sox' "
        warn_msg += "is installed or this format is not recognized.\n"
        warn_msg += "Audio file was saved as '{}'"
        assert warn_msg.format(tmp_expected_filename) == str(ae_error.value)
        ffmpef_avconv = [
            "-y",
            "-f",
            "wav",
            "-i",
            tmp_expected_filename,
            "-f",
            "ogg",
            expected_filename,
        ]
        expected_calls = [
            call(["ffmpeg"] + ffmpef_avconv),
            call(["avconv"] + ffmpef_avconv),
            call(
                [
                    "sox",
                    "-t",
                    "wav",
                    tmp_expected_filename,
                    expected_filename,
                ]
            ),
        ]
        assert patch_rsp.mock_calls == expected_calls
        region = AudioRegion.load(
            "tests/data/test_split_10HZ_mono.raw", sr=10, sw=2, ch=1
        )
        assert not saver._exported
        assert saver.data == bytes(region)


@pytest.mark.parametrize(
    "export_format",
    [
        "raw",  # raw
        "wav",  # wav
    ],
    ids=[
        "raw",
        "raw",
    ],
)
def test_AudioEventsJoinerWorker(audio_data_source, export_format):
    with TemporaryDirectory() as tmpdir:
        expected_filename = os.path.join(tmpdir, f"output.{export_format}")
        joiner = AudioEventsJoinerWorker(
            silence_duration=1.0,
            filename=expected_filename,
            export_format=export_format,
            sampling_rate=audio_data_source.sampling_rate,
            sample_width=audio_data_source.sample_width,
            channels=audio_data_source.channels,
        )

        tokenizer = TokenizerWorker(audio_data_source, observers=[joiner])
        tokenizer.start_all()
        tokenizer.join()
        joiner.join()

        output_filename = joiner.export_audio()
        expected_region = split_and_join_with_silence(
            "tests/data/test_split_10HZ_mono.raw",
            silence_duration=1.0,
            sr=10,
            sw=2,
            ch=1,
            aw=0.1,
        )
        assert output_filename == expected_filename
        assert joiner.data == bytes(expected_region)
