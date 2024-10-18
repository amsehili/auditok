import os
import subprocess
import sys
import wave
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from datetime import datetime, timedelta
from queue import Empty, Queue
from tempfile import NamedTemporaryFile
from threading import Thread

from .core import make_silence, split
from .exceptions import AudioEncodingError, AudioEncodingWarning
from .io import _guess_audio_format
from .util import AudioReader, make_duration_formatter

_STOP_PROCESSING = "STOP_PROCESSING"
_Detection = namedtuple("_Detection", "id start end duration")


def _run_subprocess(command):
    try:
        with subprocess.Popen(
            command,
            stdin=open(os.devnull, "rb"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            stdout, stderr = proc.communicate()
            return proc.returncode, stdout, stderr
    except Exception as exc:
        err_msg = "Couldn't export audio using command: '{}'".format(command)
        raise AudioEncodingError(err_msg) from exc


class Worker(Thread, metaclass=ABCMeta):
    def __init__(self, timeout=0.5, logger=None):
        self._timeout = timeout
        self._logger = logger
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
        self._logger.info(message)

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


class TokenizerWorker(Worker, AudioReader):
    def __init__(self, reader, observers=None, logger=None, **kwargs):
        self._observers = observers if observers is not None else []
        self._reader = reader
        kwargs["input"] = self
        self._audio_region_gen = split(**kwargs)
        self._detections = []
        self._log_format = "[DET]: Detection {0.id} (start: {0.start:.3f}, "
        self._log_format += "end: {0.end:.3f}, duration: {0.duration:.3f})"
        Worker.__init__(self, timeout=0.2, logger=logger)

    def _process_message(self):
        pass

    @property
    def detections(self):
        return self._detections

    @property
    def reader(self):
        return self._reader

    def _notify_observers(self, message):
        for observer in self._observers:
            observer.send(message)

    def run(self):
        self._reader.open()
        start_processing_timestamp = datetime.now()
        for _id, audio_region in enumerate(self._audio_region_gen, start=1):
            timestamp = start_processing_timestamp + timedelta(
                seconds=audio_region.meta.start
            )
            audio_region.meta.timestamp = timestamp
            detection = _Detection(
                _id,
                audio_region.meta.start,
                audio_region.meta.end,
                audio_region.duration,
            )
            self._detections.append(detection)
            if self._logger is not None:
                message = self._log_format.format(detection)
                self._log(message)
            self._notify_observers((_id, audio_region))
        self._notify_observers(_STOP_PROCESSING)
        self._reader.close()

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


class AudioDataSaverWorker(Worker):

    def __init__(
        self,
        filename,
        export_format,
        sampling_rate,
        sample_width,
        channels,
        timeout=0.2,
    ):

        super().__init__(timeout=timeout)
        self._output_filename = filename
        self._sampling_rate = sampling_rate
        self._sample_width = sample_width
        self._channels = channels

        self._export_format = _guess_audio_format(filename, export_format)
        if self._export_format is None:
            self._export_format = "wav"
        self._init_output_stream()
        self._exported = False

    @property
    def sr(self):
        return self._sampling_rate

    @property
    def sw(self):
        return self._sample_width

    @property
    def ch(self):
        return self._channels

    def _get_non_existent_filename(self):
        filename = self._output_filename + ".wav"
        i = 0
        while os.path.exists(filename):
            i += 1
            filename = self._output_filename + "({}).wav".format(i)
        return filename

    def _init_output_stream(self):
        if self._export_format != "wav":
            self._tmp_output_filename = self._get_non_existent_filename()
        else:
            self._tmp_output_filename = self._output_filename
        self._wfp = wave.open(self._tmp_output_filename, "wb")
        self._wfp.setframerate(self.sr)
        self._wfp.setsampwidth(self.sw)
        self._wfp.setnchannels(self.ch)

    @property
    def data(self):
        with wave.open(self._tmp_output_filename, "rb") as wfp:
            return wfp.readframes(-1)

    def export_audio(self):
        try:
            self._encode_export_audio()
        except AudioEncodingError as ae_error:
            raise AudioEncodingWarning(str(ae_error)) from ae_error
        return self._output_filename

    def _encode_export_audio(self):
        if self._exported:
            return self._output_filename

        if self._export_format in ("raw", "wav"):
            if self._export_format == "raw":
                self._export_raw()
            self._exported = True
            return self._output_filename
        try:
            self._export_with_ffmpeg_or_avconv()

        except AudioEncodingError:
            try:
                self._export_with_sox()
            except AudioEncodingError as exc:
                warn_msg = "Couldn't save audio data in the desired format "
                warn_msg += "'{}'.\nEither none of 'ffmpeg', 'avconv' or 'sox' "
                warn_msg += "is installed or this format is not recognized.\n"
                warn_msg += "Audio file was saved as '{}'"
                raise AudioEncodingError(
                    warn_msg.format(
                        self._export_format, self._tmp_output_filename
                    )
                ) from exc
            else:
                self._exported = True
        else:
            self._exported = True
        return self._output_filename

    def _export_raw(self):
        with open(self._output_filename, "wb") as fp:
            fp.write(self.data)

    def _export_with_ffmpeg_or_avconv(self):
        command = [
            "-y",
            "-f",
            "wav",
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
        return stdout, stderr

    def _export_with_sox(self):
        command = [
            "sox",
            "-t",
            "wav",
            self._tmp_output_filename,
            self._output_filename,
        ]
        returncode, stdout, stderr = _run_subprocess(command)
        if returncode != 0:
            raise AudioEncodingError(stderr)
        return stdout, stderr

    def close_output(self):
        self._wfp.close()

    def __del__(self):
        self._post_process()

        if (
            (self._tmp_output_filename != self._output_filename)
            and self._exported
            and os.path.exists(self._tmp_output_filename)
        ):
            os.remove(self._tmp_output_filename)


class StreamSaverWorker(AudioDataSaverWorker):
    def __init__(
        self,
        audio_reader,
        filename,
        export_format=None,
        cache_size_sec=0.5,
        timeout=0.2,
    ):
        self._reader = audio_reader
        super().__init__(
            filename,
            export_format,
            self._reader.sr,
            self._reader.sw,
            self._reader.ch,
            timeout=timeout,
        )

        sample_size_bytes = self._reader.sw * self._reader.ch
        self._cache_size = cache_size_sec * self._reader.sr * sample_size_bytes

        self._exported = False
        self._cache = []
        self._total_cached = 0

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
        self._wfp.close()

    def _write_cached_data(self):
        if self._cache:
            data = b"".join(self._cache)
            self._wfp.writeframes(data)
            self._cache = []
            self._total_cached = 0

    def open(self):
        self._reader.open()

    def close(self):
        self._reader.close()
        self.stop()

    def rewind(self):
        # ensure compatibility with AudioReader with record=True
        pass

    @property
    def data(self):
        with wave.open(self._tmp_output_filename, "rb") as wfp:
            return wfp.readframes(-1)

    def read(self):
        data = self._reader.read()
        if data is not None:
            self.send(data)
        else:
            self.send(_STOP_PROCESSING)
        return data

    def __getattr__(self, name):
        try:
            return getattr(self._reader, name)
        except AttributeError:
            return getattr(self, name)


class AudioEventsJoinerWorker(AudioDataSaverWorker):

    def __init__(
        self,
        silence_duration,
        filename,
        export_format,
        sampling_rate,
        sample_width,
        channels,
        timeout=0.2,
    ):

        super().__init__(
            filename,
            export_format,
            sampling_rate,
            sample_width,
            channels,
            timeout,
        )

        self._silence_data = make_silence(
            silence_duration, sampling_rate, sample_width, channels
        ).data
        self._first_event = True

    def _process_message(self, message):
        _, audio_event = message
        self._write_audio_event(audio_event.data)

    def _post_process(self):
        while True:
            try:
                message = self._inbox.get_nowait()
                if message != _STOP_PROCESSING:
                    _, audio_event = message
                    self._write_audio_event(audio_event.data)
            except Empty:
                break
        self._wfp.close()

    def _write_audio_event(self, data):
        if not self._first_event:
            self._wfp.writeframes(self._silence_data)
        else:
            self._first_event = False
        self._wfp.writeframes(data)


class PlayerWorker(Worker):
    def __init__(self, player, progress_bar=False, timeout=0.2, logger=None):
        self._player = player
        self._progress_bar = progress_bar
        self._log_format = "[PLAY]: Detection {id} played"
        Worker.__init__(self, timeout=timeout, logger=logger)

    def _process_message(self, message):
        _id, audio_region = message
        if self._logger is not None:
            message = self._log_format.format(id=_id)
            self._log(message)
        audio_region.play(
            player=self._player, progress_bar=self._progress_bar, leave=False
        )


class RegionSaverWorker(Worker):
    def __init__(
        self,
        filename_format,
        audio_format=None,
        timeout=0.2,
        logger=None,
        **audio_parameters,
    ):
        self._filename_format = filename_format
        self._audio_format = audio_format
        self._audio_parameters = audio_parameters
        self._debug_format = "[SAVE]: Detection {id} saved as '{filename}'"
        Worker.__init__(self, timeout=timeout, logger=logger)

    def _process_message(self, message):
        _id, audio_region = message
        filename = self._filename_format.format(
            id=_id,
            start=audio_region.meta.start,
            end=audio_region.meta.end,
            duration=audio_region.duration,
        )
        filename = audio_region.save(
            filename, self._audio_format, **self._audio_parameters
        )
        if self._logger:
            message = self._debug_format.format(id=_id, filename=filename)
            self._log(message)


class CommandLineWorker(Worker):
    def __init__(self, command, timeout=0.2, logger=None):
        self._command = command
        Worker.__init__(self, timeout=timeout, logger=logger)
        self._debug_format = "[COMMAND]: Detection {id} command: '{command}'"

    def _process_message(self, message):
        _id, audio_region = message
        with NamedTemporaryFile(delete=False) as file:
            filename = audio_region.save(file.name, audio_format="wav")
            command = self._command.format(file=filename)
            os.system(command)
            if self._logger is not None:
                message = self._debug_format.format(id=_id, command=command)
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
        self._format_time = make_duration_formatter(time_format)
        self._timestamp_format = timestamp_format
        self.detections = []
        Worker.__init__(self, timeout=timeout)

    def _process_message(self, message):
        _id, audio_region = message
        timestamp = audio_region.meta.timestamp
        timestamp = timestamp.strftime(self._timestamp_format)
        text = self._print_format.format(
            id=_id,
            start=self._format_time(audio_region.meta.start),
            end=self._format_time(audio_region.meta.end),
            duration=self._format_time(audio_region.duration),
            timestamp=timestamp,
        )
        print(text)
