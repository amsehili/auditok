import sys
import logging
from collections import namedtuple

LOGGER_NAME = "AUDITOK_LOGGER"
KeywordArguments = namedtuple("KeywordArguments", ["io", "split"])


class TimeFormatError(Exception):
    pass


def make_kwargs(args_ns):
    if args_ns.output_main is None:
        record = args_ns.plot or (args_ns.save_image is not None)
    else:
        record = False
    try:
        use_channel = int(args_ns.use_channel)
    except ValueError:
        use_channel = args_ns.use_channel

    io_kwargs = {
        "max_read": args_ns.max_time,
        "block_dur": args_ns.analysis_window,
        "sampling_rate": args_ns.sampling_rate,
        "sample_width": args_ns.sample_width,
        "channels": args_ns.channels,
        "use_channel": use_channel,
        "input_type": args_ns.input_type,
        "output_type": args_ns.output_type,
        "large_file": args_ns.large_file,
        "frames_per_buffer": args_ns.frame_per_buffer,
        "input_device_index": args_ns.input_device_index,
        "record": record,
    }

    split_kwargs = {
        "min_dur": args_ns.min_duration,
        "max_dur": args_ns.max_duration,
        "max_silence": args_ns.max_silence,
        "drop_trailing_silence": args_ns.drop_trailing_silence,
        "strict_min_dur": args_ns.strict_min_duration,
        "energy_threshold": args_ns.energy_threshold,
    }
    return KeywordArguments(io_kwargs, split_kwargs)


def make_duration_fromatter(fmt):
    """
    Accepted format directives: %i %s %m %h
    """
    if fmt == "%S":

        def fromatter(seconds):
            return "{:.3f}".format(seconds)

    elif fmt == "%I":

        def fromatter(seconds):
            return "{0}".format(int(seconds * 1000))

    else:
        fmt = fmt.replace("%h", "{hrs:d}")
        fmt = fmt.replace("%m", "{mins:d}")
        fmt = fmt.replace("%s", "{secs:d}")
        fmt = fmt.replace("%i", "{millis:03d}")
        try:
            i = fmt.index("%")
            raise TimeFormatError(
                "Unknow time format directive '{0}'".format(fmt[i : i + 2])
            )
        except ValueError:
            pass

        def fromatter(seconds):
            millis = int(seconds * 1000)
            hrs, millis = divmod(millis, 3600000)
            mins, millis = divmod(millis, 60000)
            secs, millis = divmod(millis, 1000)
            return fmt.format(hrs=hrs, mins=mins, secs=secs, millis=millis)

    return fromatter


def make_logger(debug_stdout=False, debug_file=None):
    if not debug_stdout and debug_file is None:
        return None
    logger = logging.getLogger(LOGGER_NAME)
    if debug_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    if debug_file is not None:
        handler = logging.FileHandler(debug_file, "w")
        fmt = logging.Formatter("[%(asctime)s] | %(message)s")
        handler.setFormatter(fmt)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger
