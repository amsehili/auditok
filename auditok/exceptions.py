class DuplicateArgument(Exception):
    pass


class TooSamllBlockDuration(ValueError):
    """Raised when block_dur results in a block_size smaller than one sample"""

    def __init__(self, message, block_dur, sampling_rate):
        self.block_dur = block_dur
        self.sampling_rate = sampling_rate
        super(TooSamllBlockDuration, self).__init__(message)


class TimeFormatError(Exception):
    """Raised when duration formatting directicve is wrong"""


class EndOfProcessing(Exception):
    """Raised within command line script's main function to jump to 
    postprocessing code"""


class AudioEncodingError(Exception):
    """Raised if audio data can not be encoded in the provided format"""
