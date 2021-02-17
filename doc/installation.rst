Installation
------------

A basic version of ``auditok`` will run with standard Python (>=3.4). However,
without installing additional dependencies, ``auditok`` can only deal with audio
files in *wav* or *raw* formats. if you want more features, the following
packages are needed:

- `pydub <https://github.com/jiaaro/pydub>`_ : read audio files in popular audio formats (ogg, mp3, etc.) or extract audio from a video file.
- `pyaudio <https://people.csail.mit.edu/hubert/pyaudio>`_ : read audio data from the microphone and play audio back.
- `tqdm <https://github.com/tqdm/tqdm>`_ : show progress bar while playing audio clips.
- `matplotlib <https://matplotlib.org/stable/index.html>`_ : plot audio signal and detections.
- `numpy <https://numpy.org/>`_ : required by matplotlib. Also used for some math operations instead of standard python if available.


Install the latest stable version with pip:

.. code:: bash

    sudo pip install auditok

Install with the latest development version from github:

.. code:: bash

    pip install git+https://github.com/amsehili/auditok

or

.. code:: bash

    git clone https://github.com/amsehili/auditok.git
    cd auditok
    python setup.py install
