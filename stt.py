import os
from openai import OpenAI

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")

def transcribe_wav(client: OpenAI, wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        text = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            response_format="text",
        )
    return text if isinstance(text, str) else getattr(text, "text", str(text))
