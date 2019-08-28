#!/usr/bin/env python
# encoding: utf-8
"""
auditok.auditok -- Audio Activity Detection tool

auditok.auditok is a program that can be used for Audio/Acoustic
activity detection. It can read audio data from audio files as well
as from built-in device(s) or standard input.

@author:     Mohamed El Amine SEHILI
@copyright:  2015-2019 Mohamed El Amine SEHILI
@license:    GPL v3
@contact:    amine.sehili@gmail.com
@deffield    updated: 01 Dec 2018
"""

import sys
import os
from argparse import ArgumentParser
import time
import threading

from auditok import __version__, AudioRegion
from .util import AudioDataSource
from .io import player_for
from .cmdline_util import make_logger, make_kwargs, initialize_workers
from . import workers


__all__ = []
version = __version__
__date__ = "2015-11-23"
__updated__ = "2018-12-01"


def main(argv=None):
    program_name = os.path.basename(sys.argv[0])
    if argv is None:
        argv = sys.argv[1:]
    try:
        parser = ArgumentParser(
            prog=program_name, description="An Audio Tokenization tool"
        )
        parser.add_argument("--version", "-v", action="version", version=version)
        group = parser.add_argument_group("Input-Output options")
        group.add_argument(
            "-i",
            "--input",
            dest="input",
            help="Input audio or video file. Use - for stdin "
            "[default: read from microphone using pyaudio]",
            metavar="FILE",
        )
        group.add_argument(
            "-I",
            "--input-device-index",
            dest="input_device_index",
            help="Audio device index [default: %(default)s] - only when using PyAudio",
            type=int,
            default=None,
            metavar="INT",
        )
        group.add_argument(
            "-F",
            "--audio-frame-per-buffer",
            dest="frame_per_buffer",
            help="Audio frame per buffer [default: %(default)s] - only when using PyAudio",
            type=int,
            default=1024,
            metavar="INT",
        )
        group.add_argument(
            "-t",
            "--input-type",
            dest="input_type",
            type=str,
            default=None,
            help="Input audio file type. Mandatory if file name has no extension",
            metavar="STRING",
        )
        group.add_argument(
            "-M",
            "--max-time",
            dest="max_time",
            type=float,
            default=None,
            help="Max data (in seconds) to read from microphone or file "
            "[default: read until the end of file/stream]",
            metavar="FLOAT",
        )
        group.add_argument(
            "-L",
            "--large-file",
            dest="large_file",
            action="store_true",
            default=False,
            help="Whether input file should be treated as a large file. "
            "If True, data will be read from file on demand, otherwise all "
            "audio data is loaded to memory before tokenization.",
        )
        group.add_argument(
            "-O",
            "--output-main",
            dest="output_main",
            type=str,
            default=None,
            help="Save acquired audio data to disk. If omitted no data will be saved "
            "[default: omitted]",
            metavar="FILE",
        )
        group.add_argument(
            "-o",
            "--output-tokens",
            dest="output_tokens",
            type=str,
            default=None,
            help="Output file name format for detections."
            "Use {N}, {start} and {end} to build file names,"
            "example: 'Det_{N}_{start}-{end}.wav'",
            metavar="STRING",
        )
        group.add_argument(
            "-T",
            "--output-type",
            dest="output_type",
            type=str,
            default=None,
            help="Audio type used to save detections and/or main stream. "
            "If not supplied, then it will: (1. be guessed from extension or (2. "
            "use raw format",
            metavar="STRING",
        )
        group.add_argument(
            "-u",
            "--use-channel",
            dest="use_channel",
            type=str,
            default="1",
            help="Choose channel to use from a multi-channel audio file "
            "'left' (1st channel), 'right' (2nd channel) and 'mix' "
            "(average of all channels) are accepted values. "
            "[Default: 1]",
            metavar="INT/STRING",
        )

        group = parser.add_argument_group(
            "Tokenization options", "Set tokenizer options."
        )
        group.add_argument(
            "-a",
            "--analysis-window",
            dest="analysis_window",
            default=0.01,
            type=float,
            help="Size of analysis window in seconds [default: %(default)s (10ms)]",
            metavar="FLOAT",
        )
        group.add_argument(
            "-n",
            "--min-duration",
            dest="min_duration",
            type=float,
            default=0.2,
            help="Min duration of a valid audio event in seconds [default: %(default)s]",
            metavar="FLOAT",
        )
        group.add_argument(
            "-m",
            "--max-duration",
            dest="max_duration",
            type=float,
            default=5,
            help="Max duration of a valid audio event in seconds [default: %(default)s]",
            metavar="FLOAT",
        )
        group.add_argument(
            "-s",
            "--max-silence",
            dest="max_silence",
            type=float,
            default=0.3,
            help="Max duration of a consecutive silence within a valid audio event "
            "in seconds [default: %(default)s]",
            metavar="FLOAT",
        )
        group.add_argument(
            "-d",
            "--drop-trailing-silence",
            dest="drop_trailing_silence",
            action="store_true",
            default=False,
            help="Drop trailing silence from a detection [default: keep "
            "trailing silence]",
        )
        group.add_argument(
            "-R",
            "--strict-min-duration",
            dest="strict_min_duration",
            action="store_true",
            default=False,
            help="Reject an event shorter than --min-duration even if it's "
            "adjacent to the latest valid event that reached max-duration "
            "[default: keep such events]",
        )
        group.add_argument(
            "-e",
            "--energy-threshold",
            dest="energy_threshold",
            type=float,
            default=50,
            help="Log energy threshold for detection [default: %(default)s]",
            metavar="FLOAT",
        )

        group = parser.add_argument_group(
            "Audio parameters",
            "Define audio parameters if data is read from a "
            "headerless file (raw or stdin) or you want to use "
            "different microphone parameters.",
        )
        group.add_argument(
            "-r",
            "--rate",
            dest="sampling_rate",
            type=int,
            default=16000,
            help="Sampling rate of audio data [default: %(default)s]",
            metavar="INT",
        )
        group.add_argument(
            "-c",
            "--channels",
            dest="channels",
            type=int,
            default=1,
            help="Number of channels of audio data [default: %(default)s]",
            metavar="INT",
        )
        group.add_argument(
            "-w",
            "--width",
            dest="sample_width",
            type=int,
            default=2,
            help="Number of bytes per audio sample [default: %(default)s]",
            metavar="INT",
        )

        group = parser.add_argument_group(
            "Do something with audio events",
            "Use these options to print, play or plot detections.",
        )
        group.add_argument(
            "-C",
            "--command",
            dest="command",
            type=str,
            help="Command to call when an audio detection occurs. Use $ to "
            "represent the file name to use with the command (e.g. -C "
            "'du -h $')",
            metavar="STRING",
        )
        group.add_argument(
            "-E",
            "--echo",
            dest="echo",
            action="store_true",
            default=False,
            help="Play back each detection immediately using pyaudio",
        )
        group.add_argument(
            "-B",
            "--progress-bar",
            dest="progress_bar",
            action="store_true",
            default=False,
            help="Show a progress bar when playing audio",
        )
        group.add_argument(
            "-p",
            "--plot",
            dest="plot",
            action="store_true",
            default=False,
            help="Plot and show audio signal and detections (requires matplotlib)",
        )
        group.add_argument(
            "--save-image",
            dest="save_image",
            type=str,
            help="Save plotted audio signal and detections as a picture or a PDF "
            "file (requires matplotlib)",
            metavar="FILE",
        )
        group.add_argument(
            "--printf",
            dest="printf",
            type=str,
            default="{id} {start} {end}",
            help="print detections, one per line, using a user supplied format "
            "(e.g. '[{id}]: {start} -- {end}'). Available keywords are: "
            "{id}, {start}, {end}, {duration} and {timestamp} "
            "(i.e., system date and time)",
            metavar="STRING",
        )
        group.add_argument(
            "--time-format",
            dest="time_format",
            type=str,
            default="%S",
            help="format used to print {start} and {end}.[default= %(default)s]. "
            "%%S: absolute time in seconds. %%I: absolute time in ms. If at least "
            "one of (%%h, %%m, %%s, %%i) is used, convert time into hours, "
            "minutes, seconds and millis (e.g. %%h:%%m:%%s.%%i). Only supplied "
            "fields are printed. Note that %%S and %%I can only be used alone",
            metavar="STRING",
        )
        group.add_argument(
            "--timestamp-format",
            dest="timestamp_format",
            type=str,
            default="%Y/%m/%d %H:%M:%S",
            help="format used to print {timestamp}. Should be a format accepted by "
            "datetime Default %%Y/%%m/%%d %%H:%%M:%%S",
        )
        parser.add_argument(
            "-q",
            "--quiet",
            dest="quiet",
            action="store_true",
            default=False,
            help="Do not print any information about detections [default: print "
            "'id', 'start' and 'end' of each detection]",
        )
        parser.add_argument(
            "-D",
            "--debug",
            dest="debug",
            action="store_true",
            default=False,
            help="Print processing operations to STDOUT",
        )
        parser.add_argument(
            "--debug-file",
            dest="debug_file",
            type=str,
            default=None,
            help="Print processing operations to FILE",
            metavar="FILE",
        )

        args = parser.parse_args(argv)
        logger = make_logger(args.debug, args.debug_file)
        kwargs = make_kwargs(args)
        reader, observers = initialize_workers(args, logger=logger, **kwargs.io)
        tokenizer_worker = workers.TokenizerWorker(
            reader, observers, logger=logger, **kwargs.split
        )
        tokenizer_worker.start_all()

        while True:
            time.sleep(1)
            if len(threading.enumerate()) == 1:
                raise workers.EndOfProcessing

    except (KeyboardInterrupt, workers.EndOfProcessing):
        if tokenizer_worker is not None:
            tokenizer_worker.stop_all()
            if args.output_main is not None:
                reader.save_stream()
            if args.plot or args.save_image is not None:
                from .plotting import plot_detections

                reader.rewind()
                main_region = AudioRegion(
                    reader.data, sr=reader.sr, sw=reader.sw, channels=reader.ch
                )
                detections = (
                    (det.start, det.end) for det in tokenizer_worker.audio_regions
                )
                plot_detections(
                    main_region,
                    reader.sr,
                    detections,
                    show=True,
                    save_as=args.save_image,
                )
        return 0


if __name__ == "__main__":
    sys.exit(main(None))
