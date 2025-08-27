import os
from openai import OpenAI

TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

def tts_wav_bytes(client: OpenAI, text: str) -> bytes:
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
    )
    return speech.read() if hasattr(speech, "read") else bytes(speech)