"""
September 2015
@author: Amine SEHILI <amine.sehili@gmail.com>
"""

from abc import ABCMeta, abstractmethod
import math
from array import array
from io import Rewindable, from_file, BufferAudioSource, PyAudioSource


try:
    import numpy
    _WITH_NUMPY = True
except ImportError as e:
    _WITH_NUMPY = False
    

__all__ = ["DataSource", "DataValidator", "StringDataSource", "ADSFactory", "AudioEnergyValidator"]
    

class DataSource():
    __metaclass__ = ABCMeta
    """
    Base class for objects passed to `StreamTokenizer.tokenize`.
    Subclasses should implement a `read` method.
    
    """
    
    @abstractmethod
    def read(self):
        """ Read a piece of data read from this source.
            If no more data is available, return None.
        """
    
    
class DataValidator():
    __metaclass__ = ABCMeta
    """
    Base class for a validator object used by `StreamTokenizer` to check
    if read data is valid.
    Subclasses should implement `is_valid` method.
    """
    
    @abstractmethod
    def is_valid(self, data):
        """
        Check whether `data` is valid
        """

class StringDataSource(DataSource):
    """
    A class that represent a `DataSource` as a string buffer.
    Each call to `read` returns on character and moves one step forward.
    If the end of the buffer is reached, `read` returns None. 
    """
     
    def __init__(self, data):
        """
        Parameters
        ----------
        `data` : 
            a basestring object.
        """
        
        self._data = None
        self._current = 0
        self.set_data(data)
        
    
    def read(self):
        if self._current >= len(self._data):
            return None
        self._current += 1
        return self._data[self._current - 1]
    
    def set_data(self, data):
        """
        Set a new data buffer.
        
        Parameters
        ----------
        `data` : 
            a basestring object.
        """
        
        if not isinstance(data, basestring):
            raise ValueError("data must an instance of basestring")
        self._data = data
        self._current = 0
        


class ADSFactory:
    """
    Factory class that makes it easy to create an `AudioDataSource` object that implements
    `DataSource` and can therefore be passed to `StreamTokenizer.tokenize`.
    
    Whether you read audio data from a file, the microphone or a memory buffer, this factory
    instantiates and returns the right `AudioDataSource` object.
    
    There are many other features you want your `AudioDataSource` object to have, such as: 
    memorize all read audio data so that you can rewind and reuse it (especially useful when 
    reading data from the microphone), read a fixed amount of data (also useful when reading 
    from the microphone), read overlapping audio frames (often needed when dosing a spectral
    analysis of data).
    
    `ADSFactory.ads` automatically creates and return object with the desired behavior according
    to the supplied keyword arguments. 
     
    
    """
    
    @staticmethod
    def ads(**kwargs):
        
        """
        Create an return an `AudioDataSource`. The type and behavior of the object is the result
        of the supplied parameters.
        
        Parameters
        ----------
        
        *No parameters* :  
           read audio data from the available built-in microphone with the default parameters.
           The returned `AudioDataSource` encapsulate an `io.PyAudioSource` object and hence 
           it accepts the next four parameters are passed to use instead of their default values.
        
        `sampling_rate` : *(int)*
            number of samples per second. Default = 16000.
        
        `sample_width` : *(int)*
            number of bytes per sample (must be in (1, 2, 4)). Default = 2
        
        `channels` : *(int)*
            number of audio channels. Default = 1 (only this value is currently accepted)  
            
        `frames_per_buffer` *(int)*:
            number of samples of PyAudio buffer. Default = 1024.
        
        `audio_source` : an `io.AudioSource` object
            read data from this audio source
            
        `filename` : *(string)*
            build an `io.AudioSource` object using this file (currently only wave format is supported)
            
        `data_buffer` : *(string)*
            build an `io.BufferAudioSource` using data in `data_buffer`. If this keyword is used,
            `sampling_rate`, `sample_width` and `channels` are passed to `io.BufferAudioSource`
            constructor and used instead of default values.
            
        `max_time` : *(float)*
             maximum time (in seconds) to read. Default behavior: read until there is no more data
             available. 
        
         
        `record` : *(bool)*
            save all read data in cache. Provide a navigable object which boasts a `rewind` method.
            Default = False.
            
          
         `block_size` : *(int)*
             number of samples to read each time the `read` method is called. Default : a block size
             that represent a window of 10ms, so for a sampling rate of 16000, the default `block_size`
             is 160, for a rate of 44100, `block_size` = 441, etc.
        
        `hop_size` : *(int)*
            determines the number of overlapping samples between two consecutive read windows. For a
            `hop_size` of value *N*, the overlap is `block_size` - *N*. Default : `hop_size` = `block_size`,
            means that there is no overlap.       
        
        """
        
        for k in kwargs.iterkeys():
            if not k in ["block_size", "hop_size", "max_time", "record", "audio_source",
                         "filename", "frames_per_buffer", "data_buffer", "filename", "sampling_rate",
                         "sample_width", "channels"]:
                raise ValueError("Invalid argument: {0}".format(k))
        
        
        if kwargs.has_key("block_size"):
            block_size = kwargs.pop("block_size")
        else:
            block_size = None
        
        if kwargs.has_key("hop_size"):
            hop_size = kwargs.pop("hop_size")
        else:
            hop_size = None
        
        if kwargs.has_key("max_time"):
            max_time = float(kwargs.pop("max_time"))
        else:
            max_time = None
        
        if kwargs.has_key("record"):
            record = kwargs.pop("record")
        else:
            record = False
        
        # Case 1: an audio source is supplied
        if kwargs.has_key("audio_source"):
            if kwargs.has_key("filename") or kwargs.has_key("data_buffer"):
                raise Warning("You should provide one of 'audio_source', 'filename' or 'data_buffer'\
                 keyword parameters. 'audio_source' will be used")
            audio_source = kwargs.pop("audio_source")
            
            
        # Case 2: a file name is supplied
        elif kwargs.has_key("filename"):
            if kwargs.has_key("data_buffer"):
                raise Warning("You should provide one of 'filename' or 'data_buffer'\
                 keyword parameters. 'filename' will be used")
            audio_source = from_file(kwargs.pop("filename"))
            
            
        # Case 3: a data_buffer is supplied 
        elif kwargs.has_key("data_buffer"):
            audio_source = BufferAudioSource(**kwargs)
            
        # Case 4: try to access native audio input
        else:
            audio_source = PyAudioSource(**kwargs)
             
        # Set default block_size to 10 ms
        if block_size is None:
            block_size = audio_source.get_sampling_rate() / 100
        
        # Instantiate base AudioDataSource  
        ads = ADSFactory.AudioDataSource(audio_source=audio_source, block_size=block_size)
        
        # Limit data to be read
        if max_time is not None:
            ads = ADSFactory.LimiterADS(ads=ads, max_time=max_time)
        
        # Record, rewind and reuse data
        if record:
            ads = ADSFactory.RecorderADS(ads=ads)
            
        # Read overlapping blocks of data
        if hop_size is not None:
            if hop_size <= 0 or  hop_size > block_size:
                raise ValueError("hop_size must be > 0 and <= block_size")
            if hop_size < block_size:
                ads = ADSFactory.OverlapADS(ads=ads, hop_size=hop_size)
        
        return ads
        
        
    class AudioDataSource(DataSource):
        
        def __init__(self, audio_source, block_size):
            
            self.audio_source = audio_source
            self.block_size = block_size
                
        def get_block_size(self):
            return self.block_size
        
        def set_block_size(self, size):
            self.block_size = size

        def get_audio_source(self):
            return self.audio_source
        
        def set_audio_source(self, audio_source):
            self.audio_source = audio_source
            
        def open(self):
            self.audio_source.open()
        
        def close(self):
            self.audio_source.close()
            
        def is_open(self):
            return self.audio_source.is_open()
        
        def get_sampling_rate(self):
            return self.audio_source.get_sampling_rate()
        
        def get_sample_width(self):
            return self.audio_source.get_sample_width()
        
        def get_channels(self):
            return self.audio_source.get_channels()
        
        
        def rewind(self):
            if isinstance(self.audio_source, Rewindable):
                self.audio_source.rewind()
            else:
                raise Exception("Audio source is not rewindable")
            
            
        
        def is_rewindable(self):
            return isinstance(self.audio_source, Rewindable)
        
            
        def read(self):
            return self.audio_source.read(self.block_size)
        
        
    
    
    class ADSDecorator(AudioDataSource):
        __metaclass__ = ABCMeta
        
        def __init__(self, ads):
            self.ads = ads
            
            self.get_block_size = self.ads.get_block_size
            self.set_block_size = self.ads.set_block_size
            self.get_audio_source = self.ads.get_audio_source
            self.open = self.ads.open
            self.close = self.ads.close
            self.is_open = self.ads.is_open
            self.get_sampling_rate = self.ads.get_sampling_rate
            self.get_sample_width = self.ads.get_sample_width
            self.get_channels = self.ads.get_channels
        
        
        def is_rewindable(self):
            return self.ads.is_rewindable
            
        def rewind(self):
            self.ads.rewind()
            self._reinit()
            
        def set_audio_source(self, audio_source):
            self.ads.set_audio_source(audio_source)
            self._reinit()
        
        def open(self):
            if not self.ads.is_open():
                self.ads.open()
                self._reinit()
            
        
        @abstractmethod
        def _reinit(self):
            pass            
        
        
    class OverlapADS(ADSDecorator):
        
        """
        Read overlapping audio frames
        """
        
        def __init__(self, ads, hop_size):
            ADSFactory.ADSDecorator.__init__(self, ads)
            
            if hop_size <= 0 or hop_size > self.get_block_size():
                raise ValueError("hop_size must be either 'None' or \
                 between 1 and block_size (both inclusive)")
            self.hop_size = hop_size
            self._actual_block_size = self.get_block_size()
            self._reinit()
            
            
            def _get_block_size():
                return self._actual_block_size
            
            #self.get_block_size = _get_block_size
            
            
            
        def _read_first_block(self):
            # For the first call, we need an entire block of size 'block_size'
            block = self.ads.read()
            if block is None:
                return None
            
            # Keep a slice of data in cache and append it in the next call
            if len(block) > self._hop_size_bytes:
                self._cache = block[self._hop_size_bytes:]
            
            # Up from the next call, we will use '_read_next_blocks'
            # and we only read 'hop_size'
            self.ads.set_block_size(self.hop_size)
            self.read = self._read_next_blocks
            
            return block
                
        def _read_next_blocks(self):
            block = self.ads.read()
            if block is None:
                return None
            
            # Append block to cache data to ensure overlap
            block = self._cache + block
            # Keep a slice of data in cache only if we have a full length block
            # if we don't that means that this is the last block
            if len(block) == self._block_size_bytes:
                self._cache = block[self._hop_size_bytes:]
            else:
                self._cache = None
                
            return block
                
                    
        def read(self):
            pass
        
        def _reinit(self):
            self._cache = None
            self.ads.set_block_size(self._actual_block_size)
            self._hop_size_bytes = self.hop_size * \
                               self.get_sample_width() * \
                               self.get_channels()
            self._block_size_bytes = self.get_block_size() * \
                               self.get_sample_width() * \
                               self.get_channels()
            self.read = self._read_first_block
     
    
    
    class LimiterADS(ADSDecorator):
        
        def __init__(self, ads, max_time):
            ADSFactory.ADSDecorator.__init__(self, ads)
            
            self.max_time = max_time
            self._reinit()
            
        def read(self):
            if self._total_read_bytes >=  self._max_read_bytes:
                return None
            block = self.ads.read()
            if block is None:
                return None
            self._total_read_bytes += len(block)
            
            if self._total_read_bytes >=  self._max_read_bytes:
                self.close()
            
            return block
                
                
        def _reinit(self):
            self._max_read_bytes = int(self.max_time  * self.get_sampling_rate()) * \
                                  self.get_sample_width() * \
                                  self.get_channels()
            self._total_read_bytes = 0
            
            
      
    
    class RecorderADS(ADSDecorator):
        
        def __init__(self, ads):
            ADSFactory.ADSDecorator.__init__(self, ads)
            
            self._reinit()
            
            
        def read(self):
            pass
        
        
        def _read_and_rec(self):
            # Read and save read data
            block = self.ads.read()
            if block is not None:
                self._cache.append(block)
            
            return block
            
            
        def _read_simple(self):
            # Read without recording
            return self.ads.read()
            
        
        def rewind(self):
            if self._record:
                # If has been recording, create a new BufferAudioSource
                # from recorded data
                dbuffer = ''.join(self._cache)
                asource = BufferAudioSource(dbuffer, self.get_sampling_rate(),
                                             self.get_sample_width(),
                                             self.get_channels())
                
                
                self.set_audio_source(asource)
                self.open()
                self._cache = []
                self._record = False
                self.read = self._read_simple
            
            else:
                self.ads.rewind()
                if not self.is_open():
                    self.open()
                    
        
        def is_rewindable(self):
            return True
        
        def _reinit(self):
            # when audio_source is replaced, start recording again
            self._record = True
            self._cache = []
            self.read = self._read_and_rec


                
            

class AudioEnergyValidator(DataValidator):
    
    
    if _WITH_NUMPY:
        
        _formats = {1: numpy.int8 , 2: numpy.int16, 4: numpy.int32}

        @staticmethod
        def _convert(signal, sample_width):
            return numpy.array(numpy.frombuffer(signal, 
                               dtype=AudioEnergyValidator._formats[sample_width]),
                               dtype=numpy.float64)
                               
            
        @staticmethod
        def _siganl_energy(signal):
                return float(numpy.dot(signal, signal)) / len(signal)
        
        @staticmethod    
        def _signal_log_energy(signal):
            energy = AudioEnergyValidator._siganl_energy(signal)
            if energy <= 0:
                return -200
            return 10. * numpy.log10(energy)
        
    else:
        
        
        _formats = {1: 'B' , 2: 'H', 4: 'I'}
        
        @staticmethod
        def _convert(signal, sample_width):
            array("d", array(AudioEnergyValidator._formats[sample_width], signal))
        
        @staticmethod
        def _siganl_energy(signal):
                energy = 0.
                for a in signal:
                    energy += a * a
                return energy / len(signal)
        
        @staticmethod    
        def _signal_log_energy(signal):
            energy = AudioEnergyValidator._siganl_energy(signal)
            if energy <= 0:
                return -200
            return 10. * math.log10(energy)
            
    
    def __init__(self, sample_width, energy_threshold=45):
        
        self.sample_width = sample_width
        self._energy_threshold = energy_threshold
        
            
    def is_valid(self, data):
        signal = AudioEnergyValidator._convert(data, self.sample_width)
        return AudioEnergyValidator._signal_log_energy(signal) >= self._energy_threshold
    
    def get_energy_threshold(self):
        return self._energy_threshold
    
    def set_energy_threshold(self, threshold):
        self._energy_threshold = threshold
        
    
    
        