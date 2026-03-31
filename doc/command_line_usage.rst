Command-line guide
==================

``auditok`` provides three subcommands: ``split`` (default, backward-compatible),
``trim``, and ``fix-pauses``. All three support file input and microphone recording.

For a summary of all options, type:

.. code:: bash

    auditok -h
    auditok split -h
    auditok trim -h
    auditok fix-pauses -h


Split audio into events
-----------------------

``split`` is the default subcommand. Both forms are equivalent:

.. code:: bash

    auditok audio.wav
    auditok split audio.wav

To adjust detection parameters:

.. code:: bash

    auditok audio.wav -e 55 -n 0.5 -m 10 -s 0.3

where:

- ``-e``, ``--energy-threshold``: energy threshold for detection, default: 50
- ``-n``, ``--min-duration``: minimum duration of a valid audio event in seconds, default: 0.2
- ``-m``, ``--max-duration``: maximum duration of a valid audio event in seconds, default: 5
- ``-s``, ``--max-silence``: maximum duration of continuous silence within a valid audio event in seconds, default: 0.3

Save detected events to individual files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``-o`` or ``--save-detections-as`` with placeholders:

.. code:: bash

    auditok audio.wav -o "{id}_{start:.3f}_{end:.3f}.wav"

Available placeholders: ``{id}`` (sequential, starting from 1), ``{start}``,
``{end}``, and ``{duration}`` (all in seconds).

Save the full audio stream
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``-O`` or ``--save-stream`` to save the complete audio data (including
silence) to disk. This is especially useful when reading from the microphone:

.. code:: bash

    auditok --save-stream output.wav

Customize output format
~~~~~~~~~~~~~~~~~~~~~~~

The ``--printf`` option controls the format of printed detection information:

.. code:: bash

    auditok audio.wav --printf "{id}: [{timestamp}] start:{start}, end:{end}, dur: {duration}"

output:

.. code:: bash

    1: [2021/02/17 20:16:02] start:1.160, end:2.390, dur: 1.230
    2: [2021/02/17 20:16:04] start:3.420, end:4.330, dur: 0.910
    3: [2021/02/17 20:16:06] start:5.010, end:5.720, dur: 0.710

The format of ``{timestamp}`` is controlled by ``--timestamp-format`` (default:
``%Y/%m/%d %H:%M:%S``) whereas that of ``{start}``, ``{end}`` and ``{duration}``
is controlled by ``--time-format`` (default: ``%S``, absolute number of seconds).

To completely disable printing detection information use ``-q``.

Play back detections
~~~~~~~~~~~~~~~~~~~~

Use ``-E`` (or ``--echo``) to immediately play each detected audio event:

.. code:: bash

    auditok -E

Alternatively, use ``-C`` to run an external command with each detection:

.. code:: bash

    auditok audio.wav -C "play -q {file}"

The ``{file}`` placeholder is replaced with a temporary WAV file containing
the detected event.

Plot detections
~~~~~~~~~~~~~~~

Use ``-p`` (or ``--plot``) to display the audio signal and detections
(requires matplotlib). Use ``--save-image`` to save the plot:

.. code:: bash

    auditok audio.wav -p --save-image "plot.png"


Trim silence
------------

Remove leading and trailing silence from audio:

.. code:: bash

    auditok trim audio.wav -o trimmed.wav

The ``-o``/``--output`` option is required for ``trim``.

Record from the microphone, trim, and save:

.. code:: bash

    auditok trim -o trimmed.wav

When recording from the microphone, a blinking indicator shows elapsed time.
Press ``Ctrl+C`` to stop recording. Use ``-q``/``--quiet`` to suppress the
recording indicator.


Normalize pauses (fix-pauses)
-----------------------------

Replace all pauses between detected events with a fixed duration of silence:

.. code:: bash

    auditok fix-pauses audio.wav -o cleaned.wav -d 0.5

Both ``-o``/``--output`` and ``-d``/``--pause-duration`` are required.

Record from the microphone, normalize pauses, and save:

.. code:: bash

    auditok fix-pauses -o cleaned.wav -d 0.5


Improving detection boundaries
------------------------------

Use ``-l``/``--max-leading-silence`` and ``-g``/``--max-trailing-silence`` to
extend detection boundaries and capture the natural attack and fade-out of speech:

.. code:: bash

    auditok audio.wav -l 0.2 -g 0.15

Values of 0.1 -- 0.3 seconds typically work well. These options are available on
all three subcommands.


Real-time microphone input
--------------------------

All subcommands read from the microphone when no input file is given:

.. code:: bash

    # Stream detection from microphone
    auditok

    # Record, trim, and save
    auditok trim -o trimmed.wav

    # Record, normalize pauses, and save
    auditok fix-pauses -o cleaned.wav -d 0.5

Reading from the microphone requires
`sounddevice <https://python-sounddevice.readthedocs.io/>`_.


Read audio from an external program
------------------------------------

You can pipe audio from an external program such as `sox`:

.. code:: bash

    rec -q -t raw -r 16000 -c 1 -b 16 -e signed - | auditok -

When reading from standard input, the same audio parameters must be set for
both the source program and ``auditok``:

+-----------------+------------+------------------+-----------------------+
| Audio parameter | sox option | `auditok` option | `auditok` default     |
+=================+============+==================+=======================+
| Sampling rate   | -r         | -r               |                 16000 |
+-----------------+------------+------------------+-----------------------+
| Sample width    | -b (bits)  | -w (bytes)       |                     2 |
+-----------------+------------+------------------+-----------------------+
| Channels        | -c         | -c               |                     1 |
+-----------------+------------+------------------+-----------------------+
| Encoding        | -e         | NA               | always a signed int   |
+-----------------+------------+------------------+-----------------------+


Common options reference
------------------------

.. code:: text

    -e, --energy-threshold     Detection threshold [default: 50]
    -n, --min-duration         Minimum event duration in seconds [default: 0.2]
    -m, --max-duration         Maximum event duration (split only) [default: 5]
    -s, --max-silence          Max silence within an event [default: 0.3]
    -l, --max-leading-silence  Silence to retain before events [default: 0]
    -g, --max-trailing-silence Trailing silence to keep [default: all]
    -r, --rate                 Sampling rate [default: 16000]
    -c, --channels             Number of channels [default: 1]
    -w, --width                Bytes per sample [default: 2]
    -q, --quiet                Suppress output
    -D, --debug                Debug mode
