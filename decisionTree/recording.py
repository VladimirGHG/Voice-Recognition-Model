import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

freq = 44100
duration = 5
device_id = 24

recording = sd.rec(int(duration * freq), 
                   samplerate=freq, channels=2,
                   device=device_id)

sd.wait()

wv.write("decisionTree/recording1.wav", recording, freq, sampwidth=2)
