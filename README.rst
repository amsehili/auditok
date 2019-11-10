
.. image:: doc/figures/auditok-logo.png
    :align: center

``auditok`` is an **Audio Activity Detection** tool that can process online data (read from an audio device or from standard input) as well as audio files. It can be used as a command line program and offers an easy to use API.


.. image:: https://travis-ci.org/amsehili/auditok.svg?branch=master
    :target: https://travis-ci.org/amsehili/auditok

.. image:: https://readthedocs.org/projects/auditok/badge/?version=latest
    :target: http://auditok.readthedocs.org/en/latest/?badge=latest
    :alt: Documentation Status

Basic example
-------------

.. code:: python

    from auditok import split

    # split returns a generator of AudioRegion objects
    audio_regions = split("audio.wav")
    for region in audio_regions:
        region.play(progress_bar=True)
        filename = region.save("/tmp/region_{meta.start:.3f}.wav")
        print("region saved as: {}".format(filename))


Example using `AudioRegion`
--------------------------

.. code:: python

    from auditok import AudioRegion
    region = AudioRegion.load("audio.wav")
    regions = region.split_and_plot() # or just region.splitp()


ouptut figure:

.. image:: doc/figures/example_1.png
