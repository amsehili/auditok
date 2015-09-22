AUDIo TOKenizer
===============

`auditok` is an **Audio Activity Detection** library that can be used with online data (i.e. microphone) or with audio files.

Requirements
------------
`auditok` uses [PyAudio](http://people.csail.mit.edu/hubert/pyaudio/) for audio acquisition and playback.
If installed, numpy  will be privileged for math operations on vectors.

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

Finally, this in this demo `auditok` is used to remove tailing and leading silence from an audio file:

    python demos/audio_trim_demo.py

Documentation
-------------

Check out  a quick start and the API documentation [here](http://amsehili.github.io/auditok/pdoc/)

Contribution
------------
Contributions are very appreciated !

License
-------
`auditok` is published under the GNU General Public License Version 3.

Author
------
Amine Sehili (<amine.sehili@gmail.com>)
