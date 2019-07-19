class DuplicateArgument(Exception):
    pass


class TooSamllBlockDuration(ValueError):
    """Raised when block_dur results in a block_size smaller than one sample"""

    def __init__(self, message, block_dur, sampling_rate):
        self.block_dur = block_dur
        self.sampling_rate = sampling_rate
        super(TooSamllBlockDuration, self).__init__(message)
