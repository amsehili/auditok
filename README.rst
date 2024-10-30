.. image:: doc/figures/auditok-logo.png
    :align: center

.. image:: https://github.com/amsehili/auditok/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/amsehili/auditok/actions/workflows/ci.yml/
    :alt: Build Status

.. image:: https://readthedocs.org/projects/auditok/badge/?version=latest
    :target: http://auditok.readthedocs.org/en/latest/?badge=latest
    :alt: Documentation Status

``auditok`` is an **Audio Activity Detection** tool that processes online data
(from an audio device or standard input) and audio files. It can be used via the command line or through its API.

Full documentation is available on `Read the Docs <https://auditok.readthedocs.io/en/latest/>`_.

Installation
------------

``auditok`` requires Python 3.7+.

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

Basic example
-------------

Here's a simple example of using `auditok` to detect audio events:

.. code:: python

    import auditok

    # `split` returns a generator of AudioRegion objects
    audio_events = auditok.split(
        "audio.wav",
        min_dur=0.2,     # Minimum duration of a valid audio event in seconds
        max_dur=4,       # Maximum duration of an event
        max_silence=0.3, # Maximum tolerated silence duration within an event
        energy_threshold=55 # Detection threshold
    )

    for i, r in enumerate(audio_events):
        # AudioRegions returned by `split` have defined 'start' and 'end' attributes
        print(f"Event {i}: {r.start:.3f}s -- {r.end:.3f}")

        # Play the audio event
        r.play(progress_bar=True)

        # Save the event with start and end times in the filename
        filename = r.save("event_{start:.3f}-{end:.3f}.wav")
        print(f"Event saved as: {filename}")

Example output:

.. code:: bash

    Event 0: 0.700s -- 1.400s
    Event saved as: event_0.700-1.400.wav
    Event 1: 3.800s -- 4.500s
    Event saved as: event_3.800-4.500.wav
    Event 2: 8.750s -- 9.950s
    Event saved as: event_8.750-9.950.wav
    Event 3: 11.700s -- 12.400s
    Event saved as: event_11.700-12.400.wav
    Event 4: 15.050s -- 15.850s
    Event saved as: event_15.050-15.850.wav

Split and plot
--------------

Visualize the audio signal with detected events:

.. code:: python

    import auditok
    region = auditok.load("audio.wav") # Returns an AudioRegion object
    regions = region.split_and_plot(...) # Or simply use `region.splitp()`

Example output:

.. image:: doc/figures/example_1.png

Split an audio stream and re-join (glue) audio events with silence
------------------------------------------------------------------

The following detects audio events within an audio stream, then insert
1 second of silence between them to create an audio with pauses.

.. code:: python

    # Create a 1-second silent audio region
    # Audio parameters must match the original stream
    from auditok import split, make_silence
    silence = make_silence(duration=1,
                           sampling_rate=16000,
                           sample_width=2,
                           channels=1)
    events = split("audio.wav")
    audio_with_pauses = silence.join(events)

Alternatively, use `split_and_join_with_silence`:

.. code:: python

    from auditok import split_and_join_with_silence
    audio_with_pauses = split_and_join_with_silence(silence_duration=1, input="audio.wav")

Limitations
-----------

The detection algorithm is based on audio signal energy. While it performs well
in low-noise environments (e.g., podcasts, language lessons, or quiet recordings),
performance may drop in noisy settings. Additionally, the algorithm does not
distinguish between speech and other sounds, so it is not suitable for Voice
Activity Detection in multi-sound environments.

License
-------

MIT.
