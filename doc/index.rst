auditok -- Audio Activity Detection
====================================

.. image:: https://img.shields.io/pypi/v/auditok.svg
    :target: https://pypi.org/project/auditok/
    :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/auditok.svg
    :target: https://pypi.org/project/auditok/
    :alt: Python versions

.. image:: https://github.com/amsehili/auditok/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/amsehili/auditok/actions/workflows/ci.yml/
    :alt: Build Status

.. image:: https://codecov.io/github/amsehili/auditok/graph/badge.svg?token=0rwAqYBdkf
    :target: https://codecov.io/github/amsehili/auditok

**auditok** is a lightweight audio activity detection library for Python. It
splits audio streams into events by thresholding signal energy (no models or
training data required).

Use it for voice activity detection, silence removal, audio segmentation,
or any task where you need to find "where the sound is" in an audio stream.


.. toctree::
    :caption: Getting started
    :maxdepth: 3

    installation
    examples

.. toctree::
    :caption: Command-line guide
    :maxdepth: 3

    command_line_usage

.. toctree::
    :caption: API Reference
    :maxdepth: 3

    audio
    core


License
-------

MIT.
