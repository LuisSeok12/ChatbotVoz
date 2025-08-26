from dotenv import load_dotenv
import os
from openai import OpenAI
import soundfile as sf
import sounddevice as sd
import io

VOICE = os.getenv("OPENAI_TTS_VOICE", "ballad")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")

def plat_wav_bytes(wav_bytes: bytes):
    data , samplerate = sf.read(io.BytesIO(wav_bytes), dtype='float32')
    sd.play(data, samplerate)
    sd.wait()

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Defina OPENAI_API_KEY no arquivo .env")
    
    client = OpenAI()  

    texto = "Olá, este é um teste de síntese de voz usando o modelo GPT-4o-mini-tts."

    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=VOICE,
        input=texto,
        response_format="wav",
        instructions="O sotaque deve ser brasileiro e neutro."
    )

    wav_bytes = speech.read() if hasattr(speech, 'read') else (
        speech if isinstance(speech, (bytes, bytearray)) else bytes(speech)
    )

    plat_wav_bytes(wav_bytes)

    with open("output.wav", "wb") as f:
        f.write(wav_bytes) 
    
    print("Áudio salvo em output.wav")

if __name__ == "__main__":
    main()