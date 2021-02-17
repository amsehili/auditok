Loading audio data
------------------

Audio data is loaded with the :func:`load` function which can read from audio
files, the microphone or use raw audio data.

From a file
===========

If the first argument of :func:`load` is a string, it should be a path to an audio
file.

.. code:: python

    import auditok
    region = auditok.load("audio.ogg")

If input file contains a raw (headerless) audio data, passing `audio_format="raw"`
and other audio parameters (`sampling_rate`, `sample_width` and `channels`) is
mandatory. In the following example we pass audio parameters with their short
names:

.. code:: python

    region = auditok.load("audio.dat",
                          audio_format="raw",
                          sr=44100, # alias for `sampling_rate`
                          sw=2      # alias for `sample_width`
                          ch=1      # alias for `channels`
                          )

From a `bytes` object
=====================

If the type of the first argument `bytes`, it's interpreted as raw audio data:

.. code:: python

    sr = 16000
    sw = 2
    ch = 1
    data = b"\0" * sr * sw * ch
    region = auditok.load(data, sr=sr, sw=sw, ch=ch)
    print(region)

output:

.. code:: bash

    AudioRegion(duration=1.000, sampling_rate=16000, sample_width=2, channels=1)

From the microphone
===================

If the first argument is `None`, :func:`load` will try to read data from the
microphone. Audio parameters, as well as the `max_read` parameter are mandatory:


.. code:: python

    sr = 16000
    sw = 2
    ch = 1
    five_sec_audio = load(None, sr=sr, sw=sw, ch=ch, max_read=5)
    print(five_sec_audio)

output:

.. code:: bash

    AudioRegion(duration=5.000, sampling_rate=16000, sample_width=2, channels=1)


Skip part of audio data
=======================

If the `skip` parameter is > 0, :func:`load` will skip that leading amount of audio
data:

.. code:: python

    import auditok
    region = auditok.load("audio.ogg", skip=2) # skip the first 2 seconds

This argument must be 0 when reading from the microphone.


Basic split example
-------------------

In the following we'll use the :func:`split` function to tokenize an audio file,
requiring that valid audio events be at least 0.2 second long, at most 4 seconds
long and contain a maximum of 0.3 second of continuous silence. Limiting the size
of detected events to 4 seconds means that an event of, say, 9.5 seconds will
be returned as two 4-second events plus a third 1.5-second event. Moreover, a
valid event might contain many *silences* as far as none of them exceeds 0.3
second.

:func:`split` returns a generator of :class:`AudioRegion`. An :class:`AudioRegion`
can be played, saved, repeated (i.e., multiplied by an integer) and concatenated
with another region (see examples below). Notice that :class:`AudioRegion` objects
returned by :func:`split` have a ``start`` a ``stop`` information stored in
their meta data that can be accessed like `object.meta.start`.

.. code:: python

    import auditok

    # split returns a generator of AudioRegion objects
    audio_regions = auditok.split(
        "audio.wav",
        min_dur=0.2,     # minimum duration of a valid audio event in seconds
        max_dur=4,       # maximum duration of an event
        max_silence=0.3, # maximum duration of tolerated continuous silence within an event
        energy_threshold=55 # threshold of detection
    )

    for i, r in enumerate(audio_regions):

        # Regions returned by `split` have 'start' and 'end' metadata fields
        print("Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))

        # play detection
        # r.play(progress_bar=True)

        # region's metadata can also be used with the `save` method
        # (no need to explicitly specify region's object and `format` arguments)
        filename = r.save("region_{meta.start:.3f}-{meta.end:.3f}.wav")
        print("region saved as: {}".format(filename))

output example:

.. code:: bash

    Region 0: 0.700s -- 1.400s
    region saved as: region_0.700-1.400.wav
    Region 1: 3.800s -- 4.500s
    region saved as: region_3.800-4.500.wav
    Region 2: 8.750s -- 9.950s
    region saved as: region_8.750-9.950.wav
    Region 3: 11.700s -- 12.400s
    region saved as: region_11.700-12.400.wav
    Region 4: 15.050s -- 15.850s
    region saved as: region_15.050-15.850.wav


Split and plot
--------------

Visualize audio signal and detections:

.. code:: python

    import auditok
    region = auditok.load("audio.wav") # returns an AudioRegion object
    regions = region.split_and_plot(...) # or just region.splitp()

output figure:

.. image:: figures/example_1.png


Read and split data from the microphone
---------------------------------------

If the first argument of :func:`split` is None, audio data is read from the
microphone (requires `pyaudio <https://people.csail.mit.edu/hubert/pyaudio>`_):

.. code:: python

    import auditok

    sr = 16000
    sw = 2
    ch = 1
    eth = 55 # alias for energy_threshold, default value is 50

    try:
        for region in auditok.split(input=None, sr=sr, sw=sw, ch=ch, eth=eth):
            print(region)
            region.play(progress_bar=True) # progress bar requires `tqdm`
    except KeyboardInterrupt:
         pass


:func:`split` will continue reading audio data until you press ``Ctrl-C``. If
you want to read a specific amount of audio data, pass the desired number of
seconds with the `max_read` argument.


Accessing recorded data after split
-----------------------------------

Using a :class:`Recorder` object you can get hold of acquired audio data:


.. code:: python

    import auditok

    sr = 16000
    sw = 2
    ch = 1
    eth = 55 # alias for energy_threshold, default value is 50

    rec = auditok.Recorder(input=None, sr=sr, sw=sw, ch=ch)

    try:
        for region in auditok.split(rec, sr=sr, sw=sw, ch=ch, eth=eth):
            print(region)
            region.play(progress_bar=True) # progress bar requires `tqdm`
    except KeyboardInterrupt:
         pass

    rec.rewind()
    full_audio = load(rec.data, sr=sr, sw=sw, ch=ch)
    # alternatively you can use
    full_audio = auditok.AudioRegion(rec.data, sr, sw, ch)


:class:`Recorder` also accepts a `max_read` argument.

Working with AudioRegions
-------------------------

The following are a couple of interesting operations you can do with
:class:`AudioRegion` objects.


Basic region information
========================

.. code:: python

    import auditok
    region = auditok.load("audio.wav")
    len(region) # number of audio samples int the regions, one channel considered
    region.duration # duration in seconds
    region.sampling_rate # alias `sr`
    region.sample_width # alias `sw`
    region.channels # alias `ch`


Concatenate regions
===================

.. code:: python

    import auditok
    region_1 = auditok.load("audio_1.wav")
    region_2 = auditok.load("audio_2.wav")
    region_3 = region_1 + region_2

Particularly useful if you want to join regions returned by :func:`split`:

.. code:: python

    import auditok
    regions = auditok.load("audio.wav").split()
    gapless_region = sum(regions)

Repeat a region
===============

Multiply by a positive integer:

.. code:: python

    import auditok
    region = auditok.load("audio.wav")
    region_x3 = region * 3

Split one region into N regions of equal size
=============================================

Divide by a positive integer (this has nothing to do with silence-based
tokenization):

.. code:: python

    import auditok
    region = auditok.load("audio.wav")
    regions = regions / 5
    assert sum(regions) == region

Note that if no perfect division is possible, the last region might be a bit
shorter than the previous N-1 regions.

Slice a region by samples, seconds or milliseconds
==================================================

Slicing an :class:`AudioRegion` can be interesting in many situations. You can for
example remove a fixed-size portion of audio data from the beginning or from the
end of a region or crop a region by an arbitrary amount as a data augmentation
strategy.

The most accurate way to slice an `AudioRegion` is to use indices that
directly refer to raw audio samples. In the following example, assuming that the
sampling rate of audio data is 16000, you can extract a 5-second region from
main region, starting from the 20th second as follows:

.. code:: python

    import auditok
    region = auditok.load("audio.wav")
    start = 20 * 16000
    stop = 25 * 16000
    five_second_region = region[start:stop]

This allows you to practically start and stop at any audio sample within the region.
Just as with a `list` you can omit one of `start` and `stop`, or both. You can
also use negative indices:

.. code:: python

    import auditok
    region = auditok.load("audio.wav")
    start = -3 * region.sr # `sr` is an alias of `sampling_rate`
    three_last_seconds = region[start:]

While slicing by raw samples is flexible, slicing with temporal indices is more
intuitive. You can do so by accessing the ``millis`` or ``seconds`` views of an
`AudioRegion` (or their shortcut alias `ms` and `sec` or `s`).

With the ``millis`` view:

.. code:: python

    import auditok
    region = auditok.load("audio.wav")
    five_second_region = region.millis[5000:10000]

or with the ``seconds`` view:

.. code:: python

    import auditok
    region = auditok.load("audio.wav")
    five_second_region = region.seconds[5:10]

``seconds`` indices can also be floats:

.. code:: python

    import auditok
    region = auditok.load("audio.wav")
    five_second_region = region.seconds[2.5:7.5]

Get arrays of audio samples
===========================

If `numpy` is not installed, the `samples` attributes is a list of audio samples
arrays (standard `array.array` objects), one per channels. If numpy is installed,
`samples` is a 2-D `numpy.ndarray` where the fist dimension is the channel
and the second is the the sample.

.. code:: python

    import auditok
    region = auditok.load("audio.wav")
    samples = region.samples
    assert len(samples) == region.channels


If `numpy` is not installed you can use:

.. code:: python

    import numpy as np
    region = auditok.load("audio.wav")
    samples = np.asarray(region)
    assert len(samples.shape) == 2
