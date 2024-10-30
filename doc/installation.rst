Installation
------------

**Dependencies**

The following dependencies are required by ``auditok`` and will be installed automatically:

- `numpy <https://numpy.org/>`_: Used for signal processing.
- `pydub <https://github.com/jiaaro/pydub>`_: to read audio files in popular formats (e.g., ogg, mp3) or extract audio from video files.
- `pyaudio <https://people.csail.mit.edu/hubert/pyaudio>`_: to read audio data from the microphone and play audio back.
- `tqdm <https://github.com/tqdm/tqdm>`_: to display a progress bar while playing audio clips.
- `matplotlib <https://matplotlib.org/stable/index.html>`_: to plot audio signal and detections.

``auditok`` requires Python 3.7 or higher.

To install the latest stable version, use pip:

.. code:: bash

    sudo pip install auditok

To install the latest development version from GitHub:

.. code:: bash

    pip install git+https://github.com/amsehili/auditok

Alternatively, clone the repository and install it manually:

.. code:: bash

    git clone https://github.com/amsehili/auditok.git
    cd auditok
    python setup.py install
