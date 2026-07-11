.. image:: https://raw.githubusercontent.com/amsehili/auditok/0cef3df7e8064707a7f3624669b3b838cb60523b/doc/figures/auditok-logo.png
    :align: center

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

.. image:: https://readthedocs.org/projects/auditok/badge/?version=latest
    :target: http://auditok.readthedocs.org/en/latest/?badge=latest
    :alt: Documentation Status

**auditok** is a lightweight, dependency-free audio activity detection library
for Python. It splits audio streams into events by thresholding signal energy
(no models or training data required).

Use it for voice activity detection, silence removal, audio segmentation,
or any task where you need to find "where the sound is" in an audio stream.
It works with files, microphone input, and streams, supports mono and
multi-channel audio, and runs from a few lines of Python or the command line.

Full documentation is available on `Read the Docs <https://auditok.readthedocs.io/en/latest/>`_.

Installation
------------

``auditok`` requires Python 3.8 or higher. The core library depends only on
``numpy``.

.. code:: bash

    pip install auditok

For plotting, audio playback, and progress bars:

.. code:: bash

    pip install auditok[all]

**Note:** Processing non-WAV formats (MP3, OGG, FLAC, video files, etc.)
requires `ffmpeg <https://ffmpeg.org/>`_ to be installed on your system.

API at a glance
---------------

+---------------------+-------------------------------------------------+-------------------------------------------------+
| Function            | Purpose                                         | Key parameters                                  |
+=====================+=================================================+=================================================+
| ``split()``         | Detect and yield audio events as a generator    | ``min_dur``, ``max_dur``, ``max_silence``,      |
|                     |                                                 | ``energy_threshold``                            |
+---------------------+-------------------------------------------------+-------------------------------------------------+
| ``trim()``          | Remove leading and trailing silence             | ``min_dur``, ``max_silence``,                   |
|                     |                                                 | ``energy_threshold``                            |
+---------------------+-------------------------------------------------+-------------------------------------------------+
| ``fix_pauses()``    | Normalize pauses between events to a fixed      | ``silence_duration``, ``min_dur``,              |
|                     | duration                                        | ``max_silence``, ``energy_threshold``           |
+---------------------+-------------------------------------------------+-------------------------------------------------+
| ``split_and_plot()``| Split and visualize results (matplotlib or      | split params + ``interactive``,                 |
|                     | interactive Jupyter widget)                     | ``save_as``                                     |
+---------------------+-------------------------------------------------+-------------------------------------------------+
| ``load()``          | Load audio from file, bytes, or mic into an     | ``sr``, ``sw``, ``ch``                          |
|                     | ``AudioRegion``                                 |                                                 |
+---------------------+-------------------------------------------------+-------------------------------------------------+

All functions accept file paths, raw bytes, ``AudioRegion`` objects, or ``None``
(to read from the microphone). ``split()``, ``trim()``, ``fix_pauses()``, and
``split_and_plot()`` are also available as ``AudioRegion`` methods.

Basic usage
-----------

.. code:: python

    import auditok

    # split returns a generator of AudioRegion objects
    audio_events = auditok.split(
        "audio.wav",
        min_dur=0.2,     # minimum duration of a valid audio event in seconds
        max_dur=4,       # maximum duration of an event
        max_silence=0.3, # maximum tolerated silence within an event
        energy_threshold=55 # detection threshold
    )

    for i, r in enumerate(audio_events):
        # AudioRegions returned by split have start and end attributes
        print(f"Event {i}: {r.start:.3f}s -- {r.end:.3f}s")

        # play the audio event
        r.play(progress_bar=True)

        # save the event with start and end times in the filename
        filename = r.save("event_{start:.3f}-{end:.3f}.wav")
        print(f"Event saved as: {filename}")

Example output:

.. code:: bash

    Event 0: 0.700s -- 1.400s
    Event saved as: event_0.700-1.400.wav
    Event 1: 3.800s -- 4.500s
    Event saved as: event_3.800-4.500.wav
    ...

Automatic energy threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of tuning ``energy_threshold`` by hand, let auditok estimate it
from the audio itself:

.. code:: python

    import auditok

    # estimate the threshold from the input's energy distribution
    audio_events = auditok.split("audio.wav", energy_threshold="auto")

    # or select the estimation method explicitly:
    # "otsu" (default): balanced, suited to audio with clear pauses
    # "percentile": noise floor + margin, suited to dense/far-field speech
    audio_events = auditok.split("audio.wav", validator="percentile")

    # "percentile" reads the noise floor at the 10th percentile of window
    # energies; use "pXX" to read it elsewhere (e.g., "p20" for a higher,
    # more selective threshold)
    audio_events = auditok.split("audio.wav", validator="p20")

Automatic thresholding adapts to each file's noise floor and level, so
the same code works across recordings that would otherwise need
different manual thresholds. For offline input (files, bytes,
``AudioRegion``), the whole signal is used for estimation (compressed
input is decoded only once). For live input (microphone, stdin), the
threshold is calibrated on the first seconds of the stream
(``calibration_dur``, default 3 s) and guarded by a lower bound
(``min_energy_threshold``, default 40 dB): the resolved threshold is
``max(min_energy_threshold, estimate)``, so a calibration window
containing only background noise — a PC fan, air conditioning, or a
muted microphone — cannot produce a meaningless threshold. The calibration audio is
replayed, so nothing is lost:

.. code:: python

    # detect events from the microphone with a calibrated threshold
    events = auditok.split(None, sr=16000, sw=2, ch=1, max_read=60,
                           energy_threshold="auto")

On the command line, use ``-e auto`` or ``-V otsu|percentile|pXX``.
Automatic estimation is optional — if you know a threshold that works
for your audio and setup, pass it explicitly (``energy_threshold=55`` /
``-e 55``) and no estimation takes place.

Using the WebRTC VAD as frame decider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Energy thresholding accepts any sufficiently loud audio. To detect
*speech* specifically, auditok can use the WebRTC voice activity
detector as its frame-level decider, while keeping auditok's event
machinery (``min_dur``, ``max_silence``, leading/trailing silence
handling) on top (requires ``pip install auditok[webrtcvad]``):

.. code:: python

    import auditok

    # webrtc as frame decider; mode (0-3) sets the aggressiveness:
    # 0/1 for far-field or noisy audio, 2 for clean close-talk audio
    speech_events = auditok.split("audio.wav", validator="webrtc:1")

    # full control via the validator object
    from auditok.validators import WebRTCVADValidator
    validator = WebRTCVADValidator(16000, 2, 1, mode=2, aggregation="any")
    speech_events = auditok.split("audio.wav", validator=validator)

Unlike automatic thresholding, this also works with live input
(microphone, stdin). On the command line, use ``-V webrtc`` or
``-V webrtc:2``.

Trim silence
~~~~~~~~~~~~

.. code:: python

    import auditok

    # Remove leading and trailing silence
    trimmed = auditok.trim("audio.wav", energy_threshold=55)
    trimmed.save("trimmed.wav")

Normalize pauses
~~~~~~~~~~~~~~~~

.. code:: python

    import auditok

    # Replace all pauses with exactly 0.5s of silence
    cleaned = auditok.fix_pauses("audio.wav", silence_duration=0.5)
    cleaned.save("cleaned.wav")

Improving detection boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Energy-based detection can clip the natural onset and fade-out of speech, where
the signal rises gradually from or falls back into silence. The
``max_leading_silence`` and ``max_trailing_silence`` parameters let you extend
detection boundaries to capture these transitions:

.. code:: python

    events = auditok.split(
        "audio.wav",
        max_leading_silence=0.2,   # prepend up to 200ms before each event
        max_trailing_silence=0.15, # keep up to 150ms of silence after each event
    )

Values of 0.1 -- 0.3 seconds typically work well. These parameters are available
on ``split()``, ``trim()``, ``fix_pauses()``, and their ``AudioRegion`` method
counterparts, as well as on the command line (``-l`` / ``--max-leading-silence``
and ``-g`` / ``--max-trailing-silence``).

How ``max_trailing_silence`` interacts with ``max_silence``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``max_silence`` and ``max_trailing_silence`` control two different things:

- ``max_silence`` decides **when** an event ends — it is the longest run of
  silence tolerated *inside* an event before the event boundary is closed.
- ``max_trailing_silence`` decides **how much** silence to keep at the end of
  the delivered event, as perceptual padding around the natural fade-out.

The accepted values for ``max_trailing_silence`` are:

- ``None`` (default): keep all trailing silence up to ``max_silence`` (no
  trimming, no extension).
- ``0``: drop all trailing silence.
- A value ``<= max_silence``: trim trailing silence to that duration.
- A value ``> max_silence``: once the event boundary is decided (at
  ``max_silence``), **continue collecting** silent frames past the boundary up
  to ``max_trailing_silence`` total. Collection stops early if a valid frame
  appears (in which case the current event is delivered with its accumulated
  trailing silence and a new event starts immediately from that frame, so
  separate events are *not* merged) or if the audio ends.

This decoupling is useful when you want **short, well-segmented events** but
still need enough fade-out padding to sound natural. A small ``max_silence``
keeps events tight, while a larger ``max_trailing_silence`` adds the fade-out:

.. code:: python

    events = auditok.split(
        "speech.wav",
        max_silence=0.1,           # close events on 100ms of silence
        max_trailing_silence=0.4,  # but keep up to 400ms of fade-out
    )

Split and plot
--------------

Visualize the audio signal with detected events:

.. code:: python

    import auditok

    import auditok
    audio = auditok.load("audio.wav")
    events = audio.split_and_plot(max_leading_silence=0.1,
                                  max_trailing_silence=0.1) # or region.splitp(...)

.. image:: https://raw.githubusercontent.com/amsehili/auditok/refs/heads/main/doc/figures/tokenization-result.png
    :align: center
    :alt: Split and plot example

Interactive widget in Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``interactive=True`` to ``split_and_plot`` to get an HTML5/Canvas/WebAudio
widget with clickable detection regions and inline playback:

.. code:: python

    events = audio.split_and_plot(interactive=True,
                                  max_leading_silence=0.1,
                                  max_trailing_silence=0.1)

.. image:: https://raw.githubusercontent.com/amsehili/auditok/refs/heads/main/doc/figures/tokenization-result-notebook-interactive.png
    :align: center
    :alt: interactive tokenization Jupyter notebook

Working with ``AudioRegion``
----------------------------

``AudioRegion`` is the central data structure. It wraps raw audio bytes with
metadata (sampling rate, sample width, channels) and provides a rich API
for slicing, combining, and exporting audio.

.. code:: python

    import auditok

    region = auditok.load("audio.wav")

    # Time-based slicing (returns a new AudioRegion)
    first_five_seconds = region.sec[0:5]
    middle = region.ms[1500:3000]  # milliseconds

    # Concatenation
    combined = region1 + region2

    # Repetition
    repeated = region * 3

    # Playback
    region.play(progress_bar=True)

    # Save with template placeholders
    region.save("output_{start:.3f}-{end:.3f}.wav")

    # Export as numpy array: shape (channels, samples)
    x = region.numpy()
    assert x.shape[0] == region.channels
    assert x.shape[1] == len(region)

In Jupyter notebooks, ``AudioRegion`` objects render as inline HTML5 audio
players automatically.

Command line
------------

``auditok`` provides three subcommands: ``split`` (default), ``trim``, and
``fix-pauses``. All three support file input and microphone recording.

Split audio into events
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    # Split a file (default subcommand, both forms are equivalent)
    auditok split audio.wav -e 55 -n 0.5 -m 10 -s 0.3
    # Or simply
    auditok audio.wav -e 55 -n 0.5 -m 10 -s 0.3

    # Save detected events to individual files
    auditok audio.wav -o "event_{id}_{start:.3f}-{end:.3f}.wav"

    # Stream from microphone
    auditok

Trim silence
~~~~~~~~~~~~

.. code:: bash

    # Remove leading and trailing silence
    auditok trim audio.wav -o trimmed.wav

    # Record from microphone, trim, and save
    auditok trim -o trimmed.wav

Normalize pauses
~~~~~~~~~~~~~~~~

.. code:: bash

    # Replace all pauses with 0.5s of silence
    auditok fix-pauses audio.wav -o cleaned.wav -d 0.5

    # Record from microphone, normalize pauses, and save
    auditok fix-pauses -o cleaned.wav -d 0.5

Common options
~~~~~~~~~~~~~~

.. code:: text

    -e, --energy-threshold   Detection threshold [default: 50]
    -n, --min-duration       Minimum event duration in seconds [default: 0.2]
    -m, --max-duration       Maximum event duration in seconds (split only) [default: 5]
    -s, --max-silence        Max silence within an event [default: 0.3]
    -l, --max-leading-silence  Silence to retain before events [default: 0]
    -g, --max-trailing-silence Trailing silence to keep [default: all]

Limitations
-----------

``auditok`` uses energy-based detection. It works well in low-noise environments
-- podcasts, language lessons, recordings in quiet rooms -- where the signal is
clearly above the background noise.

It does not distinguish speech from other sounds (music, claps, environmental
noise), and the energy threshold is static. Manual tuning per recording may be
needed for best results.


License
-------

MIT.
