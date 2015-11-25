[![Build Status](https://travis-ci.org/amsehili/auditok.svg?branch=master)](https://travis-ci.org/amsehili/auditok)
AUDIo TOKenizer 
===============

`auditok` is an **Audio Activity Detection** tool that can process online data (read from an audio device or from standard input) as well as audio files. It can be used as a command line program and offers an easy to use API.

The following two figures illustrate the detector output when:

1. the detector tolerates phases of silence of up to 0.3 second (300 ms) within an audio activity (also referred to as acoustic event):
![](doc/figures/figure_1.png)

2. the detector splits an audio activity event into many activities if the within silence is over 0.2 second:
![](doc/figures/figure_2.png)


Requirements
------------
`auditok` can be used with standard Python! 
However if you want more features, the following packages are needed:
- [pydub](https://github.com/jiaaro/pydub): read audio files of popular audio formats (ogg, mp3, etc.) or extract audio from a video file
- [PyAudio](http://people.csail.mit.edu/hubert/pyaudio/): read audio data from the microphone and play back detections
- `matplotlib`: plot audio signal and detections (see figures above)
- `numpy`: required by matplotlib. Also used for math operations instead of standard python if available
- Optionnaly, you can use `sox` or `parecord` for data acquisition and feed `auditok` using a pipe.


Installation
------------
    python setup.py install

Command line usage:
------------------

The first thing you want to check is perhaps how `auditok` detects your voice. If you have installed `PyAudio` just run (`Ctrl-C` to stop):

    auditok -D -E

Option `-D` means debug, whereas `-E` stands for echo, so `auditok` plays back whatever it detects.

If there are too many detections, use a higher value for energy threshold (the current version only implements a `validator` based on energy threshold. The use of spectral information is also desirable and might be part of future releases). To change the energy threshold (default: 45), use option `-e`:

    auditok -D -E -e 55

If you don't have `PyAudio`, you can use `sox` for data acquisition (`sudo apt-get install sox`):

    rec -q -t raw -r 16000 -c 1 -b 16 -e signed - | auditok -r 16000 -i -

With `-i -`,  `auditok` reads data from standard input.

`rec` and `play` are just an alias for `sox`. Doing so you won't be able to play audio detections (`-E` requires `Pyaudio`). Fortunately, `auditok` gives the possibility to call any command every time it detects an activity, passing the activity as a file to the user supplied command:

    rec -q -t raw -r 16000 -c 1 -b 16 -e signed - | auditok -i - -r 16000 -C "play -q -t raw -r 16000 -c 1 -b 16 -e signed $"
    
The `-C` option tells `auditok` to interpret its content as a command that is run whenever `auditok` detects an audio activity, replacing the `$` by a name of a temporary file into which the activity is saved as raw audio. Here we use `play` to play the activity, giving the necessary `play` arguments for raw data.

The `-C` option can be useful in many cases. Imagine a command that sends audio data over a network only if there is an audio activity and saves bandwidth during silence.

### Plot signal and detections:

use option `-p`. Requires `matplotlib` and `numpy`

### read data from file

    auditok -i input.wav ...

Install `pydub` for other audio formats.

### Limit the length of aquired data

    auditok -M 12 ...

Time is in seconds.

### Save the whole acquired audio signal

    auditok -O output.wav ...

Install `pydub` for other audio formats.


### Save each detection into a separate audio file

    auditok -o det_{N}_{start}_{end}.wav ...

You can use a free text and place `{N}`, `{start}` and `{end}` wherever you want, they will be replaced by detection number, start time and end time respectively. Another example:

    auditok -o {start}-{end}.wav ...
    
Install `pydub` for more audio formats.

Demos
-----
This code reads data from the microphone and plays back whatever it detects.

    python demos/echo.py

`echo.py` accepts two arguments: energy threshold (default=45) and duration in seconds (default=10):

    python demos/echo.py 50 15

   If only one argument is given it will be used for energy.
   
Try out this demo with an audio file (no argument is required):

    python demos/audio_tokenize_demo.py

Finally, in this demo `auditok` is used to remove tailing and leading silence from an audio file:

    python demos/audio_trim_demo.py

Documentation
-------------

Check out this [quick start](https://github.com/amsehili/auditok/blob/master/quickstart.rst) or the  [API documentation](http://amsehili.github.io/auditok/pdoc/).


Contribution
------------
Contributions are very appreciated !

License
-------
`auditok` is published under the GNU General Public License Version 3.

Author
------
Amine Sehili (<amine.sehili@gmail.com>)

