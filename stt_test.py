from dotenv import load_dotenv
import os
from openai import OpenAI

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Defina OPENAI_API_KEY no arquivo .env")
    
    stt_model = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    audio_path = "teste.wav"

    client = OpenAI()

    with open(audio_path, "rb") as f:
        text = client.audio.transcriptions.create(
            model=stt_model,
            file=f,
            response_format="text"
        )
    
    print("Transcrição:", text)

if __name__ == "__main__":
    main()