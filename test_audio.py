import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path


SampleRate = 16000
CHANNELS = 1
DTYPE = 'int16'
OUT = Path("teste.wav")

print("Gravando...")
audio = sd.rec(int(3 * SampleRate), samplerate=SampleRate, channels=CHANNELS, dtype=DTYPE)
sd.wait()
sf.write(OUT, audio, SampleRate, subtype='PCM_16')
print(f"Arquivo salvo em: {OUT.resolve()}")
print("Reproduzindo...")

data, sr = sf.read(OUT, dtype='float32')
sd.play(data, sr)
sd.wait()  
print("Pronto!")