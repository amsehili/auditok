#!/usr/bin/env python
# encoding: utf-8
"""`auditok` -- An Audio Activity Detection Tool

`auditok` is a program designed for audio or acoustic activity detection.
It supports reading audio data from various sources, including audio files,
microphones, and standard input.

@author:     Mohamed El Amine SEHILI
@copyright:  2015-2024 Mohamed El Amine SEHILI
@license:    MIT
@contact:    amine.sehili@gmail.com
@deffield    updated: 30 Oct 2024
"""

import os
import sys
import threading
import time
from argparse import ArgumentParser

from auditok import AudioRegion, __version__

from .cmdline_util import initialize_workers, make_kwargs, make_logger
from .exceptions import ArgumentError, EndOfProcessing

__all__ = []
__date__ = "2015-11-23"
__updated__ = "2024-10-30"


def main(argv=None):
    program_name = os.path.basename(sys.argv[0])
    if argv is None:
        argv = sys.argv[1:]
    try:
        parser = ArgumentParser(
            prog=program_name,
            description="auditok, an audio tokenization tool.",
        )
        parser.add_argument(
            "--version", "-v", action="version", version=__version__
        )
        group = parser.add_argument_group("Input-Output options:")
        group.add_argument(
            dest="input",
            help="Input audio or video file. Use '-' for stdin "
            "[Default: read from a microphone using PyAudio].",
            metavar="input",
            nargs="?",
            default=None,
        )
        group.add_argument(
            "-I",
            "--input-device-index",
            dest="input_device_index",
            help="Audio device index [Default: %(default)s]. "
            "Optional and only effective when using PyAudio.",
            type=int,
            default=None,
            metavar="INT",
        )
        group.add_argument(
            "-F",
            "--audio-frame-per-buffer",
            dest="frame_per_buffer",
            help="Audio frame per buffer [Default: %(default)s]. "
            "Optional and only effective when using PyAudio.",
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
            "header (requires pydub). If neither condition is met, an error "
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
            help="Join (glue) detected audio events with a specified duration "
            "of silence between them. To be used in combination with the "
            "--save-stream / -O option.",
            metavar="FLOAT",
        )
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
            "option chosen, saved audio events will have the same number of "
            "channels as the input stream. [Default: %(default)s, use all "
            "channels].",
            metavar="INT/STRING",
        )

        group = parser.add_argument_group(
            "Tokenization options:",
            "Set audio events' duration and set the threshold for detection.",
        )
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
            "-d",
            "--drop-trailing-silence",
            dest="drop_trailing_silence",
            action="store_true",
            default=False,
            help="Remove trailing silence from a detection. [Default: trailing "
            "silence is retained].",
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

        group = parser.add_argument_group(
            "Audio parameters:",
            "Set audio parameters when reading from a headerless file "
            "(raw or stdin) or when using custom microphone settings.",
        )
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

        group = parser.add_argument_group(
            "Use audio events:",
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
            help="Immediately play back a detected audio event using pyaudio.",
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

        args = parser.parse_args(argv)
        try:
            kwargs = make_kwargs(args)
        except ArgumentError as exc:
            print(exc, file=sys.stderr)
            return 1

        logger = make_logger(args.debug, args.debug_file)

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


if __name__ == "__main__":
    sys.exit(main(None))
