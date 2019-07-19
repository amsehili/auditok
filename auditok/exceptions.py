class DuplicateArgument(Exception):
    pass


class TooSamllBlockDuration(ValueError):
    """Raised when block_dur results in a block_size smaller than one sample"""
