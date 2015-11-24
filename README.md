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
- matplotlib: plot audio signal and detections (see figures above)
- numpy: required by matplotlib. Also used for math operations instead of standard python if available
- Optionnaly, you can use `sox` or `parecord` for data acquisition and feed `auditok` using a pipe.


Installation
------------
    python setup.py install

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
