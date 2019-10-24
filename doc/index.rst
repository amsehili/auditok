auditok, an AUDIo TOKenization tool
===================================

.. image:: https://travis-ci.org/amsehili/auditok.svg?branch=master
    :target: https://travis-ci.org/amsehili/auditok

.. image:: https://readthedocs.org/projects/auditok/badge/?version=latest
    :target: http://auditok.readthedocs.org/en/latest/?badge=latest
    :alt: Documentation Status

`auditok` is an **Audio Activity Detection** tool that can process online data (read from an audio device or from standard input) as well as audio files. It can be used as a command line program and offers an easy to use API.

The latest version of this documentation can be found at `Readthedocs <http://auditok.readthedocs.org/en/latest/>`_.

Requirements
------------

`auditok` can be used with standard Python!

However, if you want more features, the following packages are needed:

- `Pydub <https://github.com/jiaaro/pydub>`_ : read audio files in popular audio formats (ogg, mp3, etc.) or extract audio from a video file.

- `PyAudio <http://people.csail.mit.edu/hubert/pyaudio/>`_ : read audio data from the microphone and play back detections.

- `matplotlib <http://matplotlib.org/>`_ : plot audio signal and detections (see figures above).

- `numpy <http://www.numpy.org>`_ : required by matplotlib. Also used for math operations instead of standard python if available.

- Optionally, you can use `sox` or `[p]arecord` for data acquisition and feed `auditok` using a pipe.

Installation
------------

Install with pip:

.. code:: bash

    sudo pip install auditok

or install the latest version on Github:

.. code:: bash

    git clone https://github.com/amsehili/auditok.git
    cd auditok
    sudo python setup.py install

Getting started
---------------

.. toctree::
    :titlesonly:
    :maxdepth: 2

       Command-line Usage Guide <cmdline.rst>
       API Tutorial <apitutorial.rst>

API Reference
-------------

.. toctree::
    :maxdepth: 3

       auditok.core <core.rst>
       auditok.util <util.rst>
       auditok.io <io.rst>
       auditok.dataset <dataset.rst>

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
