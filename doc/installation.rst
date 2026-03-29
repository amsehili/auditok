Installation
------------

``auditok`` requires Python 3.8 or higher.

**Dependencies**

The only required dependency is:

- `numpy <https://numpy.org/>`_: used for signal processing.

The following are optional and can be installed as needed:

- `sounddevice <https://python-sounddevice.readthedocs.io/>`_: to read audio data from a microphone and play audio back.
- `tqdm <https://github.com/tqdm/tqdm>`_: to display a progress bar while playing audio clips.
- `matplotlib <https://matplotlib.org/stable/index.html>`_: to plot audio signals and detections.

For non-WAV audio formats (MP3, OGG, FLAC, etc.) or video files,
`ffmpeg <https://ffmpeg.org/>`_ must be installed and available on your
``PATH``.

**Install with pip**

To install the latest stable version:

.. code:: bash

    pip install auditok

To install with all optional dependencies:

.. code:: bash

    pip install auditok[all]

You can also install specific optional groups:

.. code:: bash

    pip install auditok[plot]       # matplotlib
    pip install auditok[device-io]  # sounddevice + tqdm

**Install from GitHub**

To install the latest development version:

.. code:: bash

    pip install git+https://github.com/amsehili/auditok

Or clone the repository:

.. code:: bash

    git clone https://github.com/amsehili/auditok.git
    cd auditok
    pip install .
