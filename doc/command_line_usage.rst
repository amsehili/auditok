Command-line guide
==================

``auditok`` can also be used from the command line. For information
about available parameters and descriptions, type:

.. code:: bash

    auditok -h

Below, we provide several examples covering the most common use cases.


Read audio data and detect audio events online
----------------------------------------------

To try ``auditok`` from the command line with your own voice, you’ll need to
either install `pyaudio <https://people.csail.mit.edu/hubert/pyaudio>`_ so
that ``auditok`` can read directly from the microphone, or record audio with
an external program (e.g., `sox`) and redirect its output to ``auditok``.

To read data directly from the microphone and use default parameters for audio
data and tokenization, simply type:

.. code:: bash

    auditok

This will print the **id**, **start time**, and **end time** of each detected
audio event. As mentioned above, no additional arguments were passed in the
previous command, so ``auditok`` will use its default values. The most important
arguments are:


- ``-n``, ``--min-duration``: minimum duration of a valid audio event in seconds, default: 0.2
- ``-m``, ``--max-duration``: maximum duration of a valid audio event in seconds, default: 5
- ``-s``, ``--max-silence``: maximum duration of a continuous silence within a valid audio event in seconds, default: 0.3
- ``-e``, ``--energy-threshold``: energy threshold for detection, default: 50


Read audio data with an external program
----------------------------------------
You can use an external program, such as `sox` (``sudo apt-get install sox``),
to record audio data in real-time, redirect it, and have `auditok` read the data
from standard input:

.. code:: bash

    rec -q -t raw -r 16000 -c 1 -b 16 -e signed - | auditok - -r 16000 -w 2 -c 1

Note that when reading data from standard input, the same audio parameters must
be set for both `sox` (or any other data generation/acquisition tool) and ``auditok``.
The following table provides a summary of the audio parameters:

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

Based on the table, the previous command can be run with the default parameters as:

.. code:: bash

    rec -q -t raw -r 16000 -c 1 -b 16 -e signed - | auditok -


Play back audio detections
--------------------------

Use the ``-E`` (or ``--echo``) option :

.. code:: bash

    auditok -E
    # or
    rec -q -t raw -r 16000 -c 1 -b 16 -e signed - | auditok - -E


Using ``-E`` requires `pyaudio`, if it's not installed you can use the ``-C``
(used to run an external command with detected audio event as argument):

.. code:: bash

    rec -q -t raw -r 16000 -c 1 -b 16 -e signed - | auditok - -C "play -q {file}"

Using the ``-C`` option, ``auditok`` will save a detected event to a temporary wav
file, fill the ``{file}`` placeholder with the temporary name and run the
command. In the above example we used ``-C`` to play audio data with an external
program but you can use it to run any other command.


Print out detection information
-------------------------------

By default, ``auditok`` outputs the **id**, **start**, and **end** times for each
detected audio event. The start and end values indicate the beginning and end of
the event within the input stream (file or microphone) in seconds. Below is an
example of the output in the default format:

.. code:: bash

    1 1.160 2.390
    2 3.420 4.330
    3 5.010 5.720
    4 7.230 7.800

The format of the output is controlled by the ``--printf`` option. Alongside
``{id}``, ``{start}`` and ``{end}`` placeholders, you can use ``{duration}`` and
``{timestamp}`` (system timestamp of detected event) placeholders.

Using the following format for example:

.. code:: bash

    auditok audio.wav  --printf "{id}: [{timestamp}] start:{start}, end:{end}, dur: {duration}"

the output will look like:

.. code:: bash

    1: [2021/02/17 20:16:02] start:1.160, end:2.390, dur: 1.230
    2: [2021/02/17 20:16:04] start:3.420, end:4.330, dur: 0.910
    3: [2021/02/17 20:16:06] start:5.010, end:5.720, dur: 0.710
    4: [2021/02/17 20:16:08] start:7.230, end:7.800, dur: 0.570


The format of ``{timestamp}`` is controlled by ``--timestamp-format`` (default:
`"%Y/%m/%d %H:%M:%S"`) whereas that of ``{start}``, ``{end}`` and ``{duration}``
by ``--time-format`` (default: `%S`, absolute number of seconds). A more detailed
format with ``--time-format`` using `%h` (hours), `%m` (minutes), `%s` (seconds)
and `%i` (milliseconds) directives is possible (e.g., "%h:%m:%s.%i).

To completely disable printing detection information use ``-q``.

Save detections
---------------

You can save audio events to disk as they're detected using ``-o`` or
``--save-detections-as``. To create a uniq file name for each event, you can use
``{id}``, ``{start}``, ``{end}`` and ``{duration}`` placeholders. Example:


.. code:: bash

    auditok --save-detections-as "{id}_{start}_{end}.wav"

When using ``{start}``, ``{end}``, and ``{duration}`` placeholders, it is
recommended to limit the number of decimal places for these values to 3. You
can do this with a format like:

.. code:: bash

    auditok -o "{id}_{start:.3f}_{end:.3f}.wav"


Save whole audio stream
-----------------------

When reading audio data from the microphone, you may want to save it to disk.
To do this, use the ``-O`` or ``--save-stream`` option:

.. code:: bash

    auditok --save-stream output.wav

Note that this will work even if you read data from a file on disk.


Join detected audio events with a silence of a given duration
-------------------------------------------------------------

Sometimes, you may want to detect audio events while also
creating a file that contains the same events with modified
pause durations.

To do this, use the ``-j`` or ``--join-detections`` option together
with the ``-O`` / ``--save-stream`` option. In the example below, we
read data from `input.wav` and save audio events to `output.wav`, adding
1-second pauses between them:


.. code:: bash

    auditok input.wav --join-detections 1 -O output.wav

Plot detections
---------------

Audio signal and detections can be plotted using the ``-p`` or ``--plot`` option.
You can also save plot to disk using ``--save-image``. The following example
demonstrates both:

.. code:: bash

    auditok -p --save-image "plot.png" # can also be 'pdf' or another image format

output example:

.. image:: figures/example_1.png

Plotting requires `matplotlib <https://matplotlib.org/stable/index.html>`_.
