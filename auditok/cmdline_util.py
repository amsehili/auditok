import sys
import logging


LOGGER_NAME = "AUDITOK_LOGGER"


class TimeFormatError(Exception):
    pass


def make_kwargs(args_namespace):
    kwargs = {
        "min_dur": args_namespace.min_duration,
        "max_dur": args_namespace.max_duration,
        "max_silence": args_namespace.max_silence,
        "drop_trailing_silence": args_namespace.drop_trailing_silence,
        "strict_min_length": args_namespace.strict_min_length,
        "energy_threshold": args_namespace.energy_threshold,
        "max_read_time": args_namespace.max_time,
        "analysis_window": args_namespace.analysis_window,
        "sampling_rate": args_namespace.sampling_rate,
        "sample_with": args_namespace.sample_width,
        "channels": args_namespace.channels,
        "use_channel": args_namespace.use_channel,
        "input_type": args_namespace.input_type,
        "output_type": args_namespace.output_type,
        "large_file": args_namespace.large_file,
        "frames_per_buffer": args_namespace.frame_per_buffer,
        "input_device_index": args_namespace.input_device_index,
    }
    return kwargs


def make_duration_fromatter(fmt):
    """
    Accepted format directives: %i %s %m %h
    """
    # check directives are correct

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
