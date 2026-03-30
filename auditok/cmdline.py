#!/usr/bin/env python
# encoding: utf-8
"""`auditok` -- An Audio Activity Detection Tool

`auditok` is a program designed for audio or acoustic activity detection.
It supports reading audio data from various sources, including audio files,
microphones, and standard input.

@author:     Mohamed El Amine SEHILI
@copyright:  2015-2026 Mohamed El Amine SEHILI
@license:    MIT
@contact:    amine.sehili@gmail.com
@deffield    updated: 30 Mar 2026
"""

import os
import sys
import tempfile
import threading
import time
from argparse import ArgumentParser

from auditok import AudioRegion, __version__

from .audio import fix_pauses, trim
from .cmdline_util import initialize_workers, make_kwargs, make_logger
from .exceptions import ArgumentError, EndOfProcessing

__all__ = []  # type: ignore[var-annotated]
__date__ = "2015-11-23"
__updated__ = "2026-03-30"

_SUBCOMMANDS = {"split", "trim", "fix-pauses"}


# ── Shared argument helpers ──────────────────────────────────────────


def _add_input_source_args(group):
    """Add input source configuration arguments."""
    group.add_argument(
        "-I",
        "--input-device-index",
        dest="input_device_index",
        help="Audio device index [Default: %(default)s]. "
        "Optional and only effective when reading from a microphone.",
        type=int,
        default=None,
        metavar="INT",
    )
    group.add_argument(
        "-F",
        "--audio-frame-per-buffer",
        dest="frame_per_buffer",
        help="Audio frame per buffer [Default: %(default)s]. "
        "Optional and only effective when reading from a microphone.",
        type=int,
        default=1024,
        metavar="INT",
    )
    group.add_argument(
        "-f",
        "--input-format",
        dest="input_format",
        type=str,
        default=None,
        help="Specify the input audio file format. If not provided, the "
        "format is inferred from the file extension. If the output file "
        "name lacks an extension, the format is guessed from the file "
        "header (requires ffmpeg). If neither condition is met, an error "
        "is raised.",
        metavar="STRING",
    )
    group.add_argument(
        "-M",
        "--max-read",
        dest="max_read",
        type=float,
        default=None,
        help="Maximum data (in seconds) to read from a microphone or a file"
        " [Default: read until the end of the file or stream].",
        metavar="FLOAT",
    )
    group.add_argument(
        "-L",
        "--large-file",
        dest="large_file",
        action="store_true",
        default=False,
        help="Whether the input file should be treated as a large file. "
        "If True, data will be read from file on demand, otherwise all "
        "audio data is loaded into memory before tokenization.",
    )


def _add_output_format_arg(group):
    """Add -T/--output-format argument."""
    group.add_argument(
        "-T",
        "--output-format",
        dest="output_format",
        type=str,
        default=None,
        help="Specify the audio format for saving detections and/or the "
        "main stream. If not provided, the format will be (1) inferred from"
        " the file extension or (2) default to raw format.",
        metavar="STRING",
    )


def _add_channel_arg(group):
    """Add -u/--use-channel argument."""
    group.add_argument(
        "-u",
        "--use-channel",
        dest="use_channel",
        type=str,
        default=None,
        help="Specify the audio channel to use for tokenization when the "
        "input stream is multi-channel (0 refers to the first channel). By "
        "default, this is set to None, meaning all channels are used, "
        "capturing any valid audio event from any channel. Alternatively, "
        "set this to 'mix' (or 'avg'/'average') to combine all channels "
        "into a single averaged channel for tokenization. Regardless of the"
        " option chosen, saved audio events will have the same number of "
        "channels as the input stream. [Default: %(default)s, use all "
        "channels].",
        metavar="INT/STRING",
    )


def _add_detection_args(group):
    """Add common detection arguments."""
    group.add_argument(
        "-a",
        "--analysis-window",
        dest="analysis_window",
        default=0.01,
        type=float,
        help="Specify the size of the analysis window in seconds. "
        "[Default: %(default)s (10ms)].",
        metavar="FLOAT",
    )
    group.add_argument(
        "-n",
        "--min-duration",
        dest="min_duration",
        type=float,
        default=0.2,
        help="Minimum duration of a valid audio event in seconds. "
        "[Default: %(default)s].",
        metavar="FLOAT",
    )
    group.add_argument(
        "-s",
        "--max-silence",
        dest="max_silence",
        type=float,
        default=0.3,
        help="Maximum duration of consecutive silence allowed within a "
        "valid audio event in seconds. [Default: %(default)s]",
        metavar="FLOAT",
    )
    group.add_argument(
        "-l",
        "--max-leading-silence",
        dest="max_leading_silence",
        type=float,
        default=0,
        help="Maximum duration (in seconds) of silence to retain before "
        "each detected event. Preserves the natural onset of sounds "
        "(e.g., the gradual rise of speech). A value of 0.1-0.3 seconds "
        "is typically a good choice. [Default: %(default)s]",
        metavar="FLOAT",
    )
    group.add_argument(
        "-g",
        "--max-trailing-silence",
        dest="max_trailing_silence",
        type=float,
        default=None,
        help="Maximum duration (in seconds) of trailing silence to keep "
        "at the end of each detected event. Use 0 to drop all trailing "
        "silence. When omitted, all trailing silence (up to --max-silence) "
        "is kept. [Default: %(default)s]",
        metavar="FLOAT",
    )
    group.add_argument(
        "-e",
        "--energy-threshold",
        dest="energy_threshold",
        type=float,
        default=50,
        help="Set the log energy threshold for detection. "
        "[Default: %(default)s]",
        metavar="FLOAT",
    )


def _add_audio_params(group):
    """Add audio parameter arguments."""
    group.add_argument(
        "-r",
        "--rate",
        dest="sampling_rate",
        type=int,
        default=16000,
        help="Sampling rate of audio data [Default: %(default)s].",
        metavar="INT",
    )
    group.add_argument(
        "-c",
        "--channels",
        dest="channels",
        type=int,
        default=1,
        help="Number of channels of audio data [Default: %(default)s].",
        metavar="INT",
    )
    group.add_argument(
        "-w",
        "--width",
        dest="sample_width",
        type=int,
        default=2,
        help="Number of bytes per audio sample [Default: %(default)s].",
        metavar="INT",
    )


def _add_debug_args(parser):
    """Add debug arguments."""
    parser.add_argument(
        "-D",
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Debug mode: output processing operations to STDOUT.",
    )
    parser.add_argument(
        "--debug-file",
        dest="debug_file",
        type=str,
        default=None,
        help="Save processing operations to the specified file.",
        metavar="FILE",
    )


# ── Subparser setup ──────────────────────────────────────────────────


def _setup_split_parser(subparsers):
    """Set up the 'split' subcommand parser."""
    parser = subparsers.add_parser(
        "split",
        usage="auditok split [input] [options]",
        description="Detect and split audio into individual events based on "
        "energy thresholds. This is the default subcommand.",
        help="Detect and split audio into events (default subcommand).",
    )

    group = parser.add_argument_group("Input-Output options")
    group.add_argument(
        dest="input",
        help="Input audio or video file. Use '-' for stdin "
        "[Default: read from a microphone using sounddevice].",
        metavar="input",
        nargs="?",
        default=None,
    )
    _add_input_source_args(group)
    group.add_argument(
        "-O",
        "--save-stream",
        dest="save_stream",
        type=str,
        default=None,
        help="Save read audio data (from a file or a microphone) to a file."
        " If omitted, no audio data will be saved.",
        metavar="FILE",
    )
    group.add_argument(
        "-o",
        "--save-detections-as",
        dest="save_detections_as",
        type=str,
        default=None,
        help="Specify the file name format to save detected events. You can "
        "use the following placeholders to construct the output file name: "
        "{id} (sequential, starting from 1), {start}, {end}, and {duration}. "
        "Time placeholders are in seconds. "
        "Example: 'Event_{id}{start}-{end}{duration:.3f}.wav'",
        metavar="STRING",
    )
    group.add_argument(
        "-j",
        "--join-detections",
        dest="join_detections",
        type=float,
        default=None,
        help="[Deprecated: use 'auditok fix-pauses' instead.] "
        "Join (glue) detected audio events with a specified duration "
        "of silence between them. To be used in combination with the "
        "--save-stream / -O option.",
        metavar="FLOAT",
    )
    _add_output_format_arg(group)
    _add_channel_arg(group)

    group = parser.add_argument_group(
        "Tokenization options",
        "Set audio events' duration and set the threshold for detection.",
    )
    _add_detection_args(group)
    group.add_argument(
        "-m",
        "--max-duration",
        dest="max_duration",
        type=float,
        default=5,
        help="Maximum duration of a valid audio event in seconds. "
        "[Default: %(default)s].",
        metavar="FLOAT",
    )
    group.add_argument(
        "-d",
        "--drop-trailing-silence",
        dest="drop_trailing_silence",
        action="store_true",
        default=False,
        help="[Deprecated: use -g/--max-trailing-silence 0 instead.] "
        "Remove trailing silence from a detection. Ignored if "
        "-g/--max-trailing-silence is also provided. "
        "[Default: trailing silence is retained].",
    )
    group.add_argument(
        "-R",
        "--strict-min-duration",
        dest="strict_min_duration",
        action="store_true",
        default=False,
        help="Reject events shorter than --min-duration, even if adjacent "
        "to the most recent valid event that reached max-duration. "
        "[Default: retain such events].",
    )

    group = parser.add_argument_group(
        "Audio parameters",
        "Set audio parameters when reading from a headerless file "
        "(raw or stdin) or when using custom microphone settings.",
    )
    _add_audio_params(group)

    group = parser.add_argument_group(
        "Use audio events",
        "Use these options to print, play, or plot detected audio events.",
    )
    group.add_argument(
        "-C",
        "--command",
        dest="command",
        type=str,
        help="Provide a command to execute when an audio event is detected."
        " Use '{file}' as a placeholder for the temporary WAV file "
        "containing the event data (e.g., `-C 'du -h {file}'` to "
        "display the file size or `-C 'play -q {file}'` to play audio "
        "with sox).",
        metavar="STRING",
    )
    group.add_argument(
        "-E",
        "--echo",
        dest="echo",
        action="store_true",
        default=False,
        help="Immediately play back a detected audio event using "
        "sounddevice.",
    )
    group.add_argument(
        "-B",
        "--progress-bar",
        dest="progress_bar",
        action="store_true",
        default=False,
        help="Show a progress bar when playing audio.",
    )
    group.add_argument(
        "-p",
        "--plot",
        dest="plot",
        action="store_true",
        default=False,
        help="Plot and displays the audio signal along with detections "
        "(requires matplotlib).",
    )
    group.add_argument(
        "--save-image",
        dest="save_image",
        type=str,
        help="Save the plotted audio signal and detections as a picture "
        "or a PDF file (requires matplotlib).",
        metavar="FILE",
    )
    group.add_argument(
        "--printf",
        dest="printf",
        type=str,
        default="{id} {start} {end}",
        help="Prints information about each audio event on a new line "
        "using the specified format. The format can include text and "
        "placeholders: {id} (sequential, starting from 1), {start}, "
        "{end}, {duration}, and {timestamp}. The first three time "
        "placeholders are in seconds, with formatting controlled by the "
        "--time-format argument. {timestamp} represents the system date "
        "and time of the event, configurable with the --timestamp-format "
        "argument. Example: '[{id}]: {start} -> {end} -- {timestamp}'.",
        metavar="STRING",
    )
    group.add_argument(
        "--time-format",
        dest="time_format",
        type=str,
        default="%S",
        help="Specify the format for printing {start}, {end}, and "
        "{duration} placeholders with --printf. [Default: %(default)s]. "
        "Accepted formats are\n:"
        " - %%S: absolute time in seconds\n"
        " - %%I: absolute time in milliseconds\n"
        " - %%h, %%m, %%s, %%i: converts time into hours, minutes, seconds,"
        " and milliseconds (e.g., %%h:%%m:%%s.%%i) and only displays "
        "provided fields.\nNote that %%S and %%I can only be used "
        "independently.",
        metavar="STRING",
    )
    group.add_argument(
        "--timestamp-format",
        dest="timestamp_format",
        type=str,
        default="%Y/%m/%d %H:%M:%S",
        help="Specify the format used for printing {timestamp}. Should be "
        "a format accepted by the 'datetime' standard module. [Default: "
        "'%%Y/%%m/%%d %%H:%%M:%%S'].",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="Quiet mode: Do not display any information on the screen.",
    )
    _add_debug_args(parser)

    parser.set_defaults(func=_run_split)


def _setup_trim_parser(subparsers):
    """Set up the 'trim' subcommand parser."""
    parser = subparsers.add_parser(
        "trim",
        usage="auditok trim [input] -o output [options]",
        description="Remove leading and trailing silence from audio, "
        "keeping everything between the first and last detected events.",
        help="Remove leading and trailing silence from audio.",
    )

    group = parser.add_argument_group("Input-Output options")
    group.add_argument(
        dest="input",
        help="Input audio or video file. Use '-' for stdin "
        "[Default: read from a microphone using sounddevice].",
        metavar="input",
        nargs="?",
        default=None,
    )
    group.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=True,
        help="Output file path.",
        metavar="FILE",
    )
    _add_input_source_args(group)
    _add_output_format_arg(group)
    _add_channel_arg(group)

    group = parser.add_argument_group(
        "Tokenization options",
        "Set detection parameters for identifying audio events.",
    )
    _add_detection_args(group)

    group = parser.add_argument_group(
        "Audio parameters",
        "Set audio parameters when reading from a headerless file "
        "(raw or stdin) or when using custom microphone settings.",
    )
    _add_audio_params(group)

    _add_debug_args(parser)

    parser.set_defaults(func=_run_trim)


def _setup_fix_pauses_parser(subparsers):
    """Set up the 'fix-pauses' subcommand parser."""
    parser = subparsers.add_parser(
        "fix-pauses",
        usage="auditok fix-pauses [input] -o output -d pause_duration [options]",
        description="Normalize pauses between detected audio events to a "
        "fixed duration, removing excess silence.",
        help="Normalize pauses between audio events to a fixed duration.",
    )

    group = parser.add_argument_group("Input-Output options")
    group.add_argument(
        dest="input",
        help="Input audio or video file. Use '-' for stdin "
        "[Default: read from a microphone using sounddevice].",
        metavar="input",
        nargs="?",
        default=None,
    )
    group.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=True,
        help="Output file path.",
        metavar="FILE",
    )
    group.add_argument(
        "-d",
        "--pause-duration",
        dest="pause_duration",
        type=float,
        required=True,
        help="Duration of silence (in seconds) to insert between detected "
        "audio events.",
        metavar="FLOAT",
    )
    _add_input_source_args(group)
    _add_output_format_arg(group)
    _add_channel_arg(group)

    group = parser.add_argument_group(
        "Tokenization options",
        "Set detection parameters for identifying audio events.",
    )
    _add_detection_args(group)

    group = parser.add_argument_group(
        "Audio parameters",
        "Set audio parameters when reading from a headerless file "
        "(raw or stdin) or when using custom microphone settings.",
    )
    _add_audio_params(group)

    _add_debug_args(parser)

    parser.set_defaults(func=_run_fix_pauses)


# ── Handler functions ────────────────────────────────────────────────


def _parse_use_channel(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return value


def _build_kwargs(args):
    """Build split and audio kwargs dicts from parsed args."""
    use_channel = _parse_use_channel(args.use_channel)
    split_kw = {
        "min_dur": args.min_duration,
        "max_silence": args.max_silence,
        "max_leading_silence": args.max_leading_silence,
        "max_trailing_silence": args.max_trailing_silence,
        "energy_threshold": args.energy_threshold,
        "analysis_window": args.analysis_window,
    }
    audio_kw = {
        "sampling_rate": args.sampling_rate,
        "sample_width": args.sample_width,
        "channels": args.channels,
        "use_channel": use_channel,
        "audio_format": args.input_format,
        "large_file": args.large_file,
        "max_read": args.max_read,
        "frames_per_buffer": args.frame_per_buffer,
        "input_device_index": args.input_device_index,
    }
    return split_kw, audio_kw


def _run_split(args):
    """Execute the split subcommand (original auditok behavior)."""
    try:
        kwargs = make_kwargs(args)
    except ArgumentError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.join_detections is not None:
        print(
            "Warning: -j/--join-detections is deprecated. "
            "Use 'auditok fix-pauses' instead.",
            file=sys.stderr,
        )

    logger = make_logger(args.debug, args.debug_file)
    tokenizer_worker = None
    stream_saver = None

    try:
        stream_saver, tokenizer_worker = initialize_workers(
            logger=logger, **kwargs.split, **kwargs.io, **kwargs.miscellaneous
        )
        tokenizer_worker.start_all()

        while True:
            time.sleep(1)
            if len(threading.enumerate()) == 1:
                raise EndOfProcessing

    except (KeyboardInterrupt, EndOfProcessing):
        if tokenizer_worker is not None:
            tokenizer_worker.stop_all()

            if stream_saver is not None:
                stream_saver.join()
                try:
                    stream_saver.export_audio()
                except Exception as aee:
                    print(aee, file=sys.stderr)

            if args.plot or args.save_image is not None:
                from .plotting import plot

                reader = tokenizer_worker.reader

                reader.rewind()
                record = AudioRegion(
                    reader.data, reader.sr, reader.sw, reader.ch
                )
                detections = (
                    (det.start, det.end) for det in tokenizer_worker.detections
                )
                plot(
                    record,
                    detections=detections,
                    energy_threshold=args.energy_threshold,
                    show=True,
                    save_as=args.save_image,
                )
    return 0


def _run_trim(args):
    """Execute the trim subcommand."""
    logger = make_logger(args.debug, args.debug_file)
    split_kw, audio_kw = _build_kwargs(args)

    if args.input is not None:
        # File input: call trim() directly
        result = trim(args.input, **split_kw, **audio_kw)
        if result:
            result.save(args.output, audio_format=args.output_format)
        else:
            print("No audio activity detected.", file=sys.stderr)
        return 0

    # Mic input: worker-based approach
    from . import workers
    from .util import AudioReader

    reader = AudioReader(
        source=None,
        block_dur=split_kw["analysis_window"],
        sampling_rate=audio_kw["sampling_rate"],
        sample_width=audio_kw["sample_width"],
        channels=audio_kw["channels"],
        frames_per_buffer=audio_kw["frames_per_buffer"],
        input_device_index=audio_kw["input_device_index"],
        max_read=audio_kw["max_read"],
    )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    stream_saver = workers.StreamSaverWorker(
        reader, filename=tmp_path, export_format="wav"
    )
    stream_saver.start()

    tokenizer_worker = workers.TokenizerWorker(
        stream_saver,
        observers=[],
        logger=logger,
        max_dur=None,
        strict_min_dur=False,
        use_channel=audio_kw["use_channel"],
        **split_kw,
    )
    tokenizer_worker.start_all()

    try:
        while True:
            time.sleep(1)
            if len(threading.enumerate()) == 1:
                raise EndOfProcessing
    except (KeyboardInterrupt, EndOfProcessing):
        tokenizer_worker.stop_all()

    detections = tokenizer_worker.detections
    if detections:
        first, last = detections[0], detections[-1]
        data = stream_saver.data
        full_region = AudioRegion(
            data, stream_saver.sr, stream_saver.sw, stream_saver.ch
        )
        trimmed = full_region.sec[first.start : last.end]
        trimmed.save(args.output, audio_format=args.output_format)
    else:
        print("No audio activity detected.", file=sys.stderr)

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    return 0


def _run_fix_pauses(args):
    """Execute the fix-pauses subcommand."""
    logger = make_logger(args.debug, args.debug_file)
    split_kw, audio_kw = _build_kwargs(args)

    if args.input is not None:
        # File input: call fix_pauses() directly
        result = fix_pauses(
            args.input, args.pause_duration, **split_kw, **audio_kw
        )
        if result:
            result.save(args.output, audio_format=args.output_format)
        else:
            print("No audio activity detected.", file=sys.stderr)
        return 0

    # Mic input: worker-based approach with AudioEventsJoinerWorker
    from . import workers
    from .util import AudioReader

    reader = AudioReader(
        source=None,
        block_dur=split_kw["analysis_window"],
        sampling_rate=audio_kw["sampling_rate"],
        sample_width=audio_kw["sample_width"],
        channels=audio_kw["channels"],
        frames_per_buffer=audio_kw["frames_per_buffer"],
        input_device_index=audio_kw["input_device_index"],
        max_read=audio_kw["max_read"],
    )

    joiner = workers.AudioEventsJoinerWorker(
        silence_duration=args.pause_duration,
        filename=args.output,
        export_format=args.output_format,
        sampling_rate=reader.sampling_rate,
        sample_width=reader.sample_width,
        channels=reader.channels,
    )

    tokenizer_worker = workers.TokenizerWorker(
        reader,
        observers=[joiner],
        logger=logger,
        max_dur=None,
        strict_min_dur=False,
        use_channel=audio_kw["use_channel"],
        **split_kw,
    )
    tokenizer_worker.start_all()

    try:
        while True:
            time.sleep(1)
            if len(threading.enumerate()) == 1:
                raise EndOfProcessing
    except (KeyboardInterrupt, EndOfProcessing):
        tokenizer_worker.stop_all()

    try:
        joiner.export_audio()
    except Exception as exc:
        print(exc, file=sys.stderr)

    return 0


# ── Main entry point ─────────────────────────────────────────────────


def main(argv=None):
    program_name = os.path.basename(sys.argv[0])
    if argv is None:
        argv = sys.argv[1:]

    # Handle --version/-v before subcommand dispatch
    if argv and argv[0] in ("--version", "-v"):
        print(__version__)
        return 0

    # Bare "auditok" → implicit "split" (start recording from mic)
    # "auditok file.wav ..." → implicit "split file.wav ..."
    if not argv:
        argv = ["split"]
    elif argv[0] in ("--help", "-h"):
        pass  # Let main parser show subcommand listing
    elif argv[0] not in _SUBCOMMANDS:
        argv = ["split"] + argv

    parser = ArgumentParser(
        prog=program_name,
        description="auditok, an audio tokenization tool.",
        epilog="Run '%(prog)s <command> -h' for help on a specific command. "
        "Running '%(prog)s' without a command is equivalent to "
        "'%(prog)s split'.",
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    _setup_split_parser(subparsers)
    _setup_trim_parser(subparsers)
    _setup_fix_pauses_parser(subparsers)

    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(None))
