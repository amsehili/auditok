"""
@author: Amine SEHILI <amine.sehili@gmail.com>
September, 2015
"""

# Trim leading and trailing silence from a record

from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for, dataset
import pyaudio
import sys

"""
The  tokenizer in the following example is set up to remove the silence
that precedes the first acoustic activity or follows the last activity 
in a record. It preserves whatever it founds between the two activities.
In other words, it removes the leading and trailing silence.

Sampling rate is 44100 sample per second, we'll use an analysis window of 100 ms
(i.e. bloc_ksize == 4410)

Energy threshold is 50.

The tokenizer will start accumulating windows up from the moment it encounters
the first analysis window of an energy >= 50. ALL the following windows will be 
kept regardless of their energy. At the end of the analysis, it will drop trailing
 windows with an energy below 50.

This is an interesting example because the audio file we're analyzing contains a very
brief noise that occurs within the leading silence. We certainly do want our tokenizer 
to stop at this point and considers whatever it comes after as a useful signal.
To force the tokenizer to ignore that brief event we use two other parameters `init_min`
ans `init_max_silence`. By `init_min`=3 and `init_max_silence`=1 we tell the tokenizer
that a valid event must start with at least 3 noisy windows, between which there
is at most 1 silent window.

Still with this configuration we can get the tokenizer detect that noise as a valid event
(if it actually contains 3 consecutive noisy frames). To circummvent this we use an enough
large analysis window (here of 100 ms) to ensure that the brief noise be surrounded by a much
longer silence and hence the energy of the overall analysis window will be below 50.

When using a shorter analysis window (of 10ms for instance, block_size == 441), the brief
noise contributes more to energy calculation which yields an energy of over 50 for the window.
Again we can deal with this situation by using a higher energy threshold (55 for example)
 
"""

try:
   # record = True so that we'll be able to rewind the source.
   asource = ADSFactory.ads(filename=dataset.was_der_mensch_saet_mono_44100_lead_tail_silence,
             record=True, block_size=4410)
   asource.open()

   original_signal = []
   # Read the whole signal
   while True:
      w = asource.read()
      if w is None:
         break
      original_signal.append(w)

   original_signal = b''.join(original_signal)


   # rewind source
   asource.rewind()

   # Create a validator with an energy threshold of 50
   validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=50)

   # Create a tokenizer with an unlimited token length and continuous silence within a token
   # Note the DROP_TRAILING_SILENCE mode that will ensure removing trailing silence
   trimmer = StreamTokenizer(validator, min_length = 20, max_length=99999999,
                             max_continuous_silence=9999999, mode=StreamTokenizer.DROP_TRAILING_SILENCE, init_min=3, init_max_silence=1)


   tokens = trimmer.tokenize(asource)

   # Make sure we only have one token
   assert len(tokens) == 1, "Should have detected one single token"

   trimmed_signal = b''.join(tokens[0][0])

   player = player_for(asource)

   print("\n ** Playing original signal (with leading and trailing silence)...")
   player.play(original_signal)
   print("\n ** Playing trimmed signal...")
   player.play(trimmed_signal)

   player.stop()
   asource.close()

except KeyboardInterrupt:

   player.stop()
   asource.close()
   sys.exit(0)

except Exception as e:
   
   sys.stderr.write(str(e) + "\n")
   sys.exit(1)
