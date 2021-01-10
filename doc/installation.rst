Installation
------------

.. code:: bash

    pip install auditok


A basic version of ``auditok`` will run with standard Python (>=3.4). However,
without installing additional dependencies, ``auditok`` can only deal with audio
files in *wav* or *raw* formats. if you want more features, the following
packages are needed:

    - `pydub <https://github.com/jiaaro/pydub>`_ : read audio files in popular
       audio formats (ogg, mp3, etc.) or extract audio from a video file.
    - `pyaudio <http://people.csail.mit.edu/hubert/pyaudio/>`_ : read audio data
       from the microphone and play back detections.
    - `tqdm <https://github.com/tqdm/tqdm>`_ : show progress bar while playing
       audio clips.
    - `matplotlib <http://matplotlib.org/>`_ : plot audio signal and detections.
    - `numpy <http://www.numpy.org>`_ : required by matplotlib. Also used for
       some math operations instead of standard python if available.
