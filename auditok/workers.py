from __future__ import print_function
import os
import sys
from abc import ABCMeta, abstractmethod
from threading import Thread
from datetime import datetime, timedelta
from collections import namedtuple
import wave
import subprocess
from .io import _guess_audio_format
from .util import AudioDataSource

try:
    import future
    from queue import Queue, Empty
except ImportError:
    if sys.version_info >= (3, 0):
        from queue import Queue, Empty
    else:
        from Queue import Queue, Empty

from .core import split
from . import cmdline_util

_STOP_PROCESSING = "STOP_PROCESSING"
_AudioRegionMeta = namedtuple("_AudioRegionMeta", "id start end duration")


class EndOfProcessing(Exception):
    pass


class AudioEncodingError(Exception):
    pass


def _run_subprocess(command):
    try:
        with subprocess.Popen(
            command, stdin=open(os.devnull, "rb"), stdout=subprocess.PIPE
        ) as proc:
            stdout, stderr = proc.communicate()
            return proc.returncode, stdout, stderr
    except:
        err_msg = "Can not export audio with command: {}".format(command)
        raise AudioEncodingError(err_msg)


class Worker(Thread):
    def __init__(self, timeout=0.5, logger=None):
        self._timeout = timeout
        self._logger = logger
        self._start_processing_timestamp = None
        self._inbox = Queue()
        Thread.__init__(self)

    def run(self):
        while True:
            message = self._get_message()
            if message == _STOP_PROCESSING:
                break
            if message is not None:
                self._process_message(message)
        self._post_process()

    @abstractmethod
    def _process_message(self, message):
        """Process incoming messages"""

    def _post_process(self):
        pass

    def _log(self, message):
        self._logger.warning(message)

    def _stop_requested(self):
        try:
            message = self._inbox.get_nowait()
            if message == _STOP_PROCESSING:
                return True
        except Empty:
            return False

    def stop(self):
        self.send(_STOP_PROCESSING)
        self.join()

    def send(self, message):
        self._inbox.put(message)

    def _get_message(self):
        try:
            message = self._inbox.get(timeout=self._timeout)
            return message
        except Empty:
            return None


class TokenizerWorker(Worker, AudioDataSource):
    def __init__(self, reader, observers=None, logger=None, **kwargs):
        self._observers = observers if observers is not None else []
        self._reader = reader
        self._audio_region_gen = split(self, **kwargs)
        self._audio_regions = []
        self._log_format = "[DET]: Detection {id} (start: {start:.3f}, "
        self._log_format = "end: {end:.3f}, duration: {duration:.3f})"
        Worker.__init__(self, timeout=0.5, logger=logger)

    @property
    def audio_regions(self):
        return self._audio_regions

    def _notify_observers(self, message):
        for observer in self._observers:
            observer.send(message)

    def _init_start_processing_timestamp(self):
        timestamp = datetime.now()
        self._start_processing_timestamp = timestamp
        for observer in self._observers:
            observer._start_processing_timestamp = timestamp

    def run(self):
        self._reader.open()
        self._init_start_processing_timestamp()
        for _id, audio_region in enumerate(self._audio_region_gen, start=1):
            ar_meta = _AudioRegionMeta(
                _id,
                audio_region.meta.start,
                audio_region.meta.end,
                audio_region.duration,
            )
            self._audio_regions.append(ar_meta)
            if self._logger is not None:
                message = self._log_format.format(
                    id=_id,
                    start=audio_region.start,
                    end=audio_region.end,
                    duration=audio_region.duration,
                )
                self._log(
                    message + " " + str(self._start_processing_timestamp)
                )
            self._notify_observers((_id, audio_region))
        self._notify_observers(_STOP_PROCESSING)
        self._reader.close()

    def add_observer(self, observer):
        observer.start_processing_timestamp = self._start_processing_timestamp
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def start_all(self):
        for observer in self._observers:
            observer.start()
        self.start()

    def stop_all(self):
        self.stop()
        for observer in self._observers:
            observer.stop()
        self._reader.close()

    def read(self):
        if self._stop_requested():
            return None
        else:
            return self._reader.read()

    def __getattr__(self, name):
        return getattr(self._reader, name)


class StreamSaverWorker(Worker, AudioDataSource):
    def __init__(
        self,
        audio_data_source,
        filename,
        format=None,
        cache_size=16000,
        timeout=0.5,
    ):

        self._audio_data_source = audio_data_source
        self._cache_size = cache_size
        self._output_filename = filename
        self._export_format = _guess_audio_format(format, filename)
        if self._export_format != "raw":
            self._tmp_output_filename = self._output_filename + ".raw"
        else:
            self._tmp_output_filename = self._output_filename
        self._fp = open(self._tmp_output_filename, "wb")
        self._exported = False
        self._cache = []
        self._total_cached = 0
        Worker.__init__(self, timeout=timeout)

    @property
    def sr(self):
        return self._audio_data_source.sampling_rate

    @property
    def sw(self):
        return self._audio_data_source.sample_width

    @property
    def ch(self):
        return self._audio_data_source.channels

    def __del__(self):
        self._post_process()
        if (
            self._tmp_output_filename != self._output_filename
        ) and self._exported:
            os.remove(self._tmp_output_filename)

    def _process_message(self, data):
        self._cache.append(data)
        self._total_cached += len(data)
        if self._total_cached >= self._cache_size:
            self._write_cached_data()

    def _post_process(self):
        while True:
            try:
                data = self._inbox.get_nowait()
                if data != _STOP_PROCESSING:
                    self._cache.append(data)
                    self._total_cached += len(data)
            except Empty:
                break
        self._write_cached_data()
        self._fp.close()

    def _write_cached_data(self):
        if self._cache:
            data = b"".join(self._cache)
            self._fp.write(data)
            self._cache = []
            self._total_cached = 0
            self._fp.flush()

    def open(self):
        self._audio_data_source.open()

    def close(self):
        self._audio_data_source.close()
        self.stop()

    def rewind(self):
        # ensure compatibility with AudioDataSource with record=True
        pass

    @property
    def data(self):
        print("reading data")
        with open(self._tmp_output_filename, "rb") as fp:
            return fp.read()

    def save_stream(self):
        if self._export_format == "raw":
            return
        if self._export_format == "wav":
            self._export_wave()
            self._exported = True
            return
        try:
            self._export_with_ffmpeg_or_avconv()
        except AudioEncodingError:
            try:
                self._export_with_sox()
            except AudioEncodingError:
                warn_msg = "Couldn't save data in the required format '{}'"
                print(warn_msg.format(self._export_format), file=sys.stderr)
                print("Saving stream as a wave file...", file=sys.stderr)
                self._output_filename += ".wav"
                self._export_wave()
                print(
                    "Audio data saved to '{}'".format(self._output_filename),
                    file=sys.stderr,
                )
        finally:
            self._exported = True
        return self._output_filename

    def _export_wave(self):
        with open(self._tmp_output_filename, "rb") as fp:
            with wave.open(self._output_filename, "wb") as wfp:
                wfp.setframerate(self.sr)
                wfp.setsampwidth(self.sw)
                wfp.setnchannels(self.ch)
                # read blocks of 4 seconds
                block_size = self.sr * self.sw * self.ch * 4
                while True:
                    block = fp.read(block_size)
                    if not block:
                        return
                    wfp.writeframes(block)

    def _export_with_ffmpeg_or_avconv(self):
        pcm_fmt = {1: "s8", 2: "s16le", 4: "s32le"}[self.sw]
        command = [
            "-y",
            "-f",
            pcm_fmt,
            "-ar",
            str(self.sr),
            "-ac",
            str(self.ch),
            "-i",
            self._tmp_output_filename,
            "-f",
            self._export_format,
            self._output_filename,
        ]
        returncode, stdout, stderr = _run_subprocess(["ffmpeg"] + command)
        if returncode != 0:
            returncode, stdout, stderr = _run_subprocess(["avconv"] + command)
            if returncode != 0:
                raise AudioEncodingError(stderr)
        return stdout

    def _export_with_sox(self):
        command = [
            "sox",
            "-t",
            "raw",
            "-r",
            str(self.sr),
            "-c",
            str(self.ch),
            "-b",
            str(self.sw * 8),
            "-e",
            "signed",
            self._tmp_output_filename,
            self._output_filename,
        ]
        returncode, stdout, stderr = _run_subprocess(command)
        if returncode != 0:
            raise AudioEncodingError(stderr)
        return stdout

    def close_output(self):
        self._fp.close()

    def read(self):
        data = self._audio_data_source.read()
        if data is not None:
            self.send(data)
        else:
            self.send(_STOP_PROCESSING)
        return data

    def __getattr__(self, name):
        if name == "data":
            return self.data
        return getattr(self._audio_data_source, name)


class PlayerWorker(Worker):
    def __init__(self, player, progress_bar=False, timeout=0.5, logger=None):
        self._player = player
        self._progress_bar = progress_bar
        self._log_format = "[PLAY]: Detection {id} played (start:{start:.3f},"
        self._log_format += "end:{end:.3f}, dur:{duration:.3f})"
        Worker.__init__(self, timeout=timeout, logger=logger)

    def _process_message(self, message):
        _id, audio_region = message
        if self._logger is not None:
            message = self._log_format.format(
                id=_id,
                start=audio_region.start,
                end=audio_region.end,
                duration=audio_region.duration,
            )
            self._log(message)
        audio_region.play(
            player=self._player, progress_bar=self._progress_bar, leave=False
        )


class RegionSaverWorker(Worker):
    def __init__(
        self, name_format, filetype=None, timeout=0.2, logger=None, **kwargs
    ):
        self._name_format = name_format
        self._filetype = filetype
        self._audio_kwargs = kwargs
        self._debug_format = '[SAVE]: Detection {id} saved as "{filename}"'
        Worker.__init__(self, timeout=timeout, logger=logger)

    def _process_message(self, message):
        _id, audio_region = message
        filename = self._name_format.format(
            id=_id,
            start=audio_region.meta.start,
            end=audio_region.meta.end,
            duration=audio_region.duration,
        )
        filename = audio_region.save(
            filename, self._filetype, **self._audio_kwargs
        )
        if self._logger:
            message = self._debug_format.format(id=_id, filename=filename)
            self._log(message)


class PrintWorker(Worker):
    def __init__(
        self,
        print_format="{start} {end}",
        time_format="%S",
        timestamp_format="%Y/%m/%d %H:%M:%S.%f",
        timeout=0.2,
    ):

        self._print_format = print_format
        self._format_time = cmdline_util.make_duration_fromatter(time_format)
        self._timestamp_format = timestamp_format
        self.detections = []
        Worker.__init__(self, timeout=timeout)

    def _process_message(self, message):
        _id, audio_region = message
        timestamp = self._start_processing_timestamp + timedelta(
            seconds=audio_region.meta.start
        )
        timestamp = timestamp.strftime(self._timestamp_format)
        text = self._print_format.format(
            id=_id,
            start=self._format_time(audio_region.meta.start),
            end=self._format_time(audio_region.meta.end),
            duration=self._format_time(audio_region.duration),
            timestamp=timestamp,
        )
        print(text)
