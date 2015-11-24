
from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for
import pyaudio
import sys

try:

   energy_threshold = 45
   duration = 10 # seconds


   if len(sys.argv) > 1:
     energy_threshold = float(sys.argv[1])

   if len(sys.argv) > 2:
     duration = float(sys.argv[2])

   # record = True so that we'll be able to rewind the source.
   # max_time = 10: read 10 seconds from the microphone
   asource = ADSFactory.ads(record=True, max_time = duration)

   validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold = energy_threshold)
   tokenizer = StreamTokenizer(validator=validator, min_length=20, max_length=250, max_continuous_silence=30)

   player = player_for(asource)

   def echo(data, start, end):
      print("Acoustic activity at: {0}--{1}".format(start, end))
      player.play(b''.join(data))

   asource.open()

   print("\n  ** Make some noise (dur:{}, energy:{})...".format(duration, energy_threshold))

   tokenizer.tokenize(asource, callback=echo)

   asource.close()
   player.stop()

except KeyboardInterrupt:

   player.stop()
   asource.close()
   sys.exit(0)

except Exception as e:
   
   sys.stderr.write(str(e) + "\n")
   sys.exit(1)
