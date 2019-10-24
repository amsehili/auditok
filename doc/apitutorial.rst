`auditok` API Tutorial
======================

.. contents:: `Contents`
   :depth: 3


**auditok**  is a module that can be used as a generic tool for data
tokenization. Although its core motivation is **Acoustic Activity
Detection** (AAD) and extraction from audio streams (i.e. detect
where a noise/an acoustic activity occurs within an audio stream and
extract the corresponding portion of signal), it can easily be
adapted to other tasks.

Globally speaking, it can be used to extract, from a sequence of
observations, all sub-sequences that meet a certain number of
criteria in terms of:

1. Minimum length of a **valid** token (i.e. sub-sequence)
2. Maximum length of a **valid** token
3. Maximum tolerated consecutive **non-valid** observations within
   a valid token

Examples of a non-valid observation are: a non-numeric ascii symbol
if you are interested in sub-sequences of numeric symbols, or a silent
audio window (of 10, 20 or 100 milliseconds for instance) if what
interests you are audio regions made up of a sequence of "noisy"
windows (whatever kind of noise: speech, baby cry, laughter, etc.).

The most important component of `auditok` is the :class:`auditok.core.StreamTokenizer`
class. An instance of this class encapsulates a :class:`auditok.util.DataValidator` and can be
configured to detect the desired regions from a stream.
The :func:`auditok.core.StreamTokenizer.tokenize` method accepts a :class:`auditok.util.DataSource`
object that has a `read` method. Read data can be of any type accepted
by the `validator`.


As the main aim of this module is **Audio Activity Detection**,
it provides the :class:`auditok.util.ADSFactory` factory class that makes
it very easy to create an :class:`auditok.util.ADSFactory.AudioDataSource`
(a class that implements :class:`auditok.util.DataSource`) object, be that from:

- A file on the disk
- A buffer of data
- The built-in microphone (requires PyAudio)


The :class:`auditok.util.ADSFactory.AudioDataSource` class inherits from
:class:`auditok.util.DataSource` and supplies a higher abstraction level
than :class:`auditok.io.AudioSource` thanks to a bunch of handy features:

- Define a fixed-length `block_size` (alias `bs`, i.e. analysis window)
- Alternatively, use `block_dur` (duration in seconds, alias `bd`)
- Allow overlap between two consecutive analysis windows
  (if one of `hop_size` , `hs` or `hop_dur` , `hd` keywords is used and is > 0 and < `block_size`).
  This can be very important if your validator use the **spectral** information of audio data
  instead of raw audio samples.
- Limit the amount (i.e. duration) of read data (if keyword `max_time` or `mt` is used, very useful when reading data from the microphone)
- Record all read data and rewind if necessary (if keyword `record` or `rec` , also useful if you read data from the microphone and
  you want to process it many times off-line and/or save it)

See :class:`auditok.util.ADSFactory` documentation for more information.

Last but not least, the current version has only one audio window validator based on
signal energy (:class:`auditok.util.AudioEnergyValidator).

**********************************
Illustrative examples with strings
**********************************

Let us look at some examples using the :class:`auditok.util.StringDataSource` class
created for test and illustration purposes. Imagine that each character of
:class:`auditok.util.StringDataSource` data represents an audio slice of 100 ms for
example. In the following examples we will use upper case letters to represent
noisy audio slices (i.e. analysis windows or frames) and lower case letter for
silent frames.


Extract sub-sequences of consecutive upper case letters
#######################################################


We want to extract sub-sequences of characters that have:

- A minimum length of 1 (`min_length` = 1)
- A maximum length of 9999 (`max_length` = 9999)
- Zero consecutive lower case characters within them (`max_continuous_silence` = 0)

We also create the `UpperCaseChecker` with a `read` method that returns `True` if the
checked character is in upper case and `False` otherwise.

.. code:: python

    from auditok import StreamTokenizer, StringDataSource, DataValidator

    class UpperCaseChecker(DataValidator):
       def is_valid(self, frame):
          return frame.isupper()

    dsource = StringDataSource("aaaABCDEFbbGHIJKccc")
    tokenizer = StreamTokenizer(validator=UpperCaseChecker(),
                 min_length=1, max_length=9999, max_continuous_silence=0)

    tokenizer.tokenize(dsource)

The output is a list of two tuples, each contains the extracted sub-sequence and its
start and end position in the original sequence respectively:


.. code:: python


    [(['A', 'B', 'C', 'D', 'E', 'F'], 3, 8), (['G', 'H', 'I', 'J', 'K'], 11, 15)]


Tolerate up to two non-valid (lower case) letters within an extracted sequence
##############################################################################

To do so, we set `max_continuous_silence` =2:

.. code:: python


    from auditok import StreamTokenizer, StringDataSource, DataValidator

    class UpperCaseChecker(DataValidator):
       def is_valid(self, frame):
          return frame.isupper()

    dsource = StringDataSource("aaaABCDbbEFcGHIdddJKee")
    tokenizer = StreamTokenizer(validator=UpperCaseChecker(),
                 min_length=1, max_length=9999, max_continuous_silence=2)

    tokenizer.tokenize(dsource)


output:

.. code:: python

    [(['A', 'B', 'C', 'D', 'b', 'b', 'E', 'F', 'c', 'G', 'H', 'I', 'd', 'd'], 3, 16), (['J', 'K', 'e', 'e'], 18, 21)]

Notice the trailing lower case letters "dd" and "ee" at the end of the two
tokens. The default behavior of :class:`auditok.core.StreamTokenizer` is to keep the *trailing
silence* if it does not exceed `max_continuous_silence`. This can be changed
using the `StreamTokenizer.DROP_TRAILING_SILENCE` mode (see next example).

Remove trailing silence
#######################

Trailing silence can be useful for many sound recognition applications, including
speech recognition. Moreover, from the human auditory system point of view, trailing
low energy signal helps removing abrupt signal cuts.

If you want to remove it anyway, you can do it by setting `mode` to `StreamTokenizer.DROP_TRAILING_SILENCE`:

.. code:: python

    from auditok import StreamTokenizer, StringDataSource, DataValidator

    class UpperCaseChecker(DataValidator):
       def is_valid(self, frame):
          return frame.isupper()

    dsource = StringDataSource("aaaABCDbbEFcGHIdddJKee")
    tokenizer = StreamTokenizer(validator=UpperCaseChecker(),
                 min_length=1, max_length=9999, max_continuous_silence=2,
                 mode=StreamTokenizer.DROP_TRAILING_SILENCE)

    tokenizer.tokenize(dsource)

output:

.. code:: python

    [(['A', 'B', 'C', 'D', 'b', 'b', 'E', 'F', 'c', 'G', 'H', 'I'], 3, 14), (['J', 'K'], 18, 19)]



Limit the length of detected tokens
###################################


Imagine that you just want to detect and recognize a small part of a long
acoustic event (e.g. engine noise, water flow, etc.) and avoid that that
event hogs the tokenizer and prevent it from feeding the event to the next
processing step (i.e. a sound recognizer). You can do this by:

 - limiting the length of a detected token.

 and

 - using a callback function as an argument to :class:`auditok.core.StreamTokenizer.tokenize`
   so that the tokenizer delivers a token as soon as it is detected.

The following code limits the length of a token to 5:

.. code:: python

    from auditok import StreamTokenizer, StringDataSource, DataValidator

    class UpperCaseChecker(DataValidator):
       def is_valid(self, frame):
          return frame.isupper()

    dsource = StringDataSource("aaaABCDEFGHIJKbbb")
    tokenizer = StreamTokenizer(validator=UpperCaseChecker(),
                 min_length=1, max_length=5, max_continuous_silence=0)

    def print_token(data, start, end):
        print("token = '{0}', starts at {1}, ends at {2}".format(''.join(data), start, end))

    tokenizer.tokenize(dsource, callback=print_token)


output:

.. code:: python

    "token = 'ABCDE', starts at 3, ends at 7"
    "token = 'FGHIJ', starts at 8, ends at 12"
    "token = 'K', starts at 13, ends at 13"


************************
`auditok` and Audio Data
************************

In the rest of this document we will use :class:`auditok.util.ADSFactory`, :class:`auditok.util.AudioEnergyValidator`
and :class:`auditok.core.StreamTokenizer` for Audio Activity Detection demos using audio data. Before we get any
further it is worth, explaining a certain number of points.

:func:`auditok.util.ADSFactory.ads` method is used to create an :class:`auditok.util.ADSFactory.AudioDataSource`
object either from a wave file, the built-in microphone or a user-supplied data buffer. Refer to the API reference
for more information and examples on :func:`ADSFactory.ads` and :class:`AudioDataSource`.

The created :class:`AudioDataSource` object is then passed to :func:`StreamTokenizer.tokenize` for tokenization.

:func:`auditok.util.ADSFactory.ads` accepts a number of keyword arguments, of which none is mandatory.
The returned :class:`AudioDataSource` object's features and behavior can however greatly differ
depending on the passed arguments. Further details can be found in the respective method documentation.

Note however the following two calls that will create an :class:`AudioDataSource`
that reads data from an audio file and from the built-in microphone respectively.

.. code:: python

    from auditok import ADSFactory

    # Get an AudioDataSource from a file
    # use 'filename', alias 'fn' keyword argument
    file_ads = ADSFactory.ads(filename = "path/to/file/")

    # Get an AudioDataSource from the built-in microphone
    # The returned object has the default values for sampling
    # rate, sample width an number of channels. see method's
    # documentation for customized values
    mic_ads = ADSFactory.ads()

For :class:`StreamTkenizer`, parameters `min_length`, `max_length` and `max_continuous_silence`
are expressed in terms of number of frames. Each call to :func:`AudioDataSource.read` returns
one frame of data or None.

If you want a `max_length` of 2 seconds for your detected sound events and your *analysis window*
is *10 ms* long, you have to specify a `max_length` of 200 (`int(2. / (10. / 1000)) == 200`).
For a `max_continuous_silence` of *300 ms* for instance, the value to pass to StreamTokenizer is 30
(`int(0.3 / (10. / 1000)) == 30`).

Each time :class:`StreamTkenizer` calls the :func:`read` (has no argument) method of an
:class:`AudioDataSource` object, it returns the same amount of data, except if there are no more
data (returns what's left in stream or None).

This fixed-length amount of data is referred here to as **analysis window** and is a parameter of
:func:`ADSFactory.ads` method. By default :func:`ADSFactory.ads` uses an analysis window of 10 ms.

The number of samples that 10 ms of audio data contain will vary, depending on the sampling
rate of your audio source/data (file, microphone, etc.).
For a sampling rate of 16KHz (16000 samples per second), we have 160 samples for 10 ms.

You can use the `block_size` keyword (alias `bs`) to define your analysis window:

.. code:: python

    from auditok import ADSFactory

    '''
    Assume you have an audio file with a sampling rate of 16000
    '''

    # file_ads.read() will return blocks of 160 sample
    file_ads = ADSFactory.ads(filename = "path/to/file/", block_size = 160)

    # file_ads.read() will return blocks of 320 sample
    file_ads = ADSFactory.ads(filename = "path/to/file/", bs = 320)


Fortunately, you can specify the size of your analysis window in seconds, thanks to keyword `block_dur`
(alias `bd`):

.. code:: python

    from auditok import ADSFactory
    # use an analysis window of 20 ms
    file_ads = ADSFactory.ads(filename = "path/to/file/", bd = 0.02)

For :class:`StreamTkenizer`, each :func:`read` call that does not return `None` is treated as a processing
frame. :class:`StreamTkenizer` has no way to figure out the temporal length of that frame (why sould it?). So to
correctly initialize your :class:`StreamTokenizer`, based on your analysis window duration, use something like:


.. code:: python

    analysis_win_seconds = 0.01 # 10 ms
    my_ads = ADSFactory.ads(block_dur = analysis_win_seconds)
    analysis_window_ms = analysis_win_seconds * 1000

    # If you want your maximum continuous silence to be 300 ms use:
    max_continuous_silence = int(300. / analysis_window_ms)

    # which is the same as:
    max_continuous_silence = int(0.3 / (analysis_window_ms / 1000))

    # or simply:
    max_continuous_silence = 30


******************************
Examples using real audio data
******************************


Extract isolated phrases from an utterance
##########################################

We will build an :class:`auditok.util.ADSFactory.AudioDataSource` using a wave file from
the database. The file contains of isolated pronunciation of digits from 1 to 1
in Arabic as well as breath-in/out between 2 and 3. The code will play the
original file then the detected sounds separately. Note that we use an
`energy_threshold` of 65, this parameter should be carefully chosen. It depends
on microphone quality, background noise and the amplitude of events you want to
detect.

.. code:: python

    from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for, dataset

    # We set the `record` argument to True so that we can rewind the source
    asource = ADSFactory.ads(filename=dataset.one_to_six_arabic_16000_mono_bc_noise, record=True)

    validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=65)

    # Default analysis window is 10 ms (float(asource.get_block_size()) / asource.get_sampling_rate())
    # min_length=20 : minimum length of a valid audio activity is 20 * 10 == 200 ms
    # max_length=4000 :  maximum length of a valid audio activity is 400 * 10 == 4000 ms == 4 seconds
    # max_continuous_silence=30 : maximum length of a tolerated  silence within a valid audio activity is 30 * 30 == 300 ms
    tokenizer = StreamTokenizer(validator=validator, min_length=20, max_length=400, max_continuous_silence=30)

    asource.open()
    tokens = tokenizer.tokenize(asource)

    # Play detected regions back

    player = player_for(asource)

    # Rewind and read the whole signal
    asource.rewind()
    original_signal = []

    while True:
       w = asource.read()
       if w is None:
          break
       original_signal.append(w)

    original_signal = ''.join(original_signal)

    print("Playing the original file...")
    player.play(original_signal)

    print("playing detected regions...")
    for t in tokens:
        print("Token starts at {0} and ends at {1}".format(t[1], t[2]))
        data = ''.join(t[0])
        player.play(data)

    assert len(tokens) == 8


The tokenizer extracts 8 audio regions from the signal, including all isolated digits
(from 1 to 6) as well as the 2-phase respiration of the subject. You might have noticed
that, in the original file, the last three digit are closer to each other than the
previous ones. If you wan them to be extracted as one single phrase, you can do so
by tolerating a larger continuous silence within a detection:

.. code:: python

    tokenizer.max_continuous_silence = 50
    asource.rewind()
    tokens = tokenizer.tokenize(asource)

    for t in tokens:
       print("Token starts at {0} and ends at {1}".format(t[1], t[2]))
       data = ''.join(t[0])
       player.play(data)

    assert len(tokens) == 6


Trim leading and trailing silence
#################################

The  tokenizer in the following example is set up to remove the silence
that precedes the first acoustic activity or follows the last activity
in a record. It preserves whatever it founds between the two activities.
In other words, it removes the leading and trailing silence.

Sampling rate is 44100 sample per second, we'll use an analysis window of 100 ms
(i.e. block_size == 4410)

Energy threshold is 50.

The tokenizer will start accumulating windows up from the moment it encounters
the first analysis window of an energy >= 50. ALL the following windows will be
kept regardless of their energy. At the end of the analysis, it will drop trailing
windows with an energy below 50.

This is an interesting example because the audio file we're analyzing contains a very
brief noise that occurs within the leading silence. We certainly do want our tokenizer
to stop at this point and considers whatever it comes after as a useful signal.
To force the tokenizer to ignore that brief event we use two other parameters `init_min`
and `init_max_silence`. By `init_min` = 3 and `init_max_silence` = 1 we tell the tokenizer
that a valid event must start with at least 3 noisy windows, between which there
is at most 1 silent window.

Still with this configuration we can get the tokenizer detect that noise as a valid event
(if it actually contains 3 consecutive noisy frames). To circumvent this we use an enough
large analysis window (here of 100 ms) to ensure that the brief noise be surrounded by a much
longer silence and hence the energy of the overall analysis window will be below 50.

When using a shorter analysis window (of 10 ms for instance, block_size == 441), the brief
noise contributes more to energy calculation which yields an energy of over 50 for the window.
Again we can deal with this situation by using a higher energy threshold (55 for example).

.. code:: python

    from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for, dataset

    # record = True so that we'll be able to rewind the source.
    asource = ADSFactory.ads(filename=dataset.was_der_mensch_saet_mono_44100_lead_trail_silence,
             record=True, block_size=4410)
    asource.open()

    original_signal = []
    # Read the whole signal
    while True:
       w = asource.read()
       if w is None:
          break
       original_signal.append(w)

    original_signal = ''.join(original_signal)

    # rewind source
    asource.rewind()

    # Create a validator with an energy threshold of 50
    validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=50)

    # Create a tokenizer with an unlimited token length and continuous silence within a token
    # Note the DROP_TRAILING_SILENCE mode that will ensure removing trailing silence
    trimmer = StreamTokenizer(validator, min_length = 20, max_length=99999999, init_min=3, init_max_silence=1, max_continuous_silence=9999999, mode=StreamTokenizer.DROP_TRAILING_SILENCE)

    tokens = trimmer.tokenize(asource)

    # Make sure we only have one token
    assert len(tokens) == 1, "Should have detected one single token"

    trimmed_signal = ''.join(tokens[0][0])

    player = player_for(asource)

    print("Playing original signal (with leading and trailing silence)...")
    player.play(original_signal)
    print("Playing trimmed signal...")
    player.play(trimmed_signal)


Online audio signal processing
##############################

In the next example, audio data is directly acquired from the built-in microphone.
The :func:`auditok.core.StreamTokenizer.tokenize` method is passed a callback function
so that audio activities are delivered as soon as they are detected. Each detected
activity is played back using the build-in audio output device.

As mentioned before , Signal energy is strongly related to many factors such
microphone sensitivity, background noise (including noise inherent to the hardware),
distance and your operating system sound settings. Try a lower `energy_threshold`
if your noise does not seem to be detected and a higher threshold if you notice
an over detection (echo method prints a detection where you have made no noise).

.. code:: python

    from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for

    # record = True so that we'll be able to rewind the source.
    # max_time = 10: read 10 seconds from the microphone
    asource = ADSFactory.ads(record=True, max_time=10)

    validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=50)
    tokenizer = StreamTokenizer(validator=validator, min_length=20, max_length=250, max_continuous_silence=30)

    player = player_for(asource)

    def echo(data, start, end):
       print("Acoustic activity at: {0}--{1}".format(start, end))
       player.play(''.join(data))

    asource.open()

    tokenizer.tokenize(asource, callback=echo)

If you want to re-run the tokenizer after changing of one or many parameters, use the following code:

.. code:: python

    asource.rewind()
    # change energy threshold for example
    tokenizer.validator.set_energy_threshold(55)
    tokenizer.tokenize(asource, callback=echo)

In case you want to play the whole recorded signal back use:

.. code:: python

    player.play(asource.get_audio_source().get_data_buffer())


************
Contributing
************

**auditok** is on `GitHub <https://github.com/amsehili/auditok>`_. You're welcome to fork it and contribute.


Amine SEHILI <amine.sehili@gmail.com>
September 2015

*******
License
*******

This package is published under GNU GPL Version 3.
