import os
import time
import queue
import tempfile
from contextlib import contextmanager
import numpy as np
import sounddevice as sd
import soundfile as sf 
from dotenv import load_dotenv
from openai import OpenAI

SAMPLERATE = 16000
CHANNELS = 1
DTYPE = 'int16'

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")

def record_until_enter() -> np.ndarray:
    print("Pressione Enter para iniciar a gravação e Enter novamente para parar...")
    input()
    print("Gravando... Pressione Enter para parar.")
    
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            pass
        q.put(indata.copy())
    
    stream = sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype=DTYPE, callback=callback)
    stream.start()

    try:
        input()
    finally:
        stream.stop()
        stream.close()
    
    chunks = []
    while not q.empty():
        chunks.append(q.get()) 
    if not chunks:
        return np.empty((0, CHANNELS), dtype=DTYPE)
    audio = np.concatenate(chunks, axis=0)
    print(f"Gravação finalizada. Duração: {len(audio) / SAMPLERATE:.2f} segundos.")
    return audio

@contextmanager
def temp_wav_path(prefix="user_", suffix=".wav"):
    path = os.path.join(tempfile.gettempdir(), f"{prefix}{int(time.time()*1000)}{suffix}")
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def transcribe_wav(client: OpenAI, wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        text = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            response_format="text"
        )
    
    return text if isinstance(text, str) else getattr(text, 'text', str(text))

def chat_response(client: OpenAI, user_text: str, history: list) -> str:
    messages = [
        {"role": "system", "content": "Você é um assistente de voz objetivo e educado. Responda em PT-BR."},
        *history[-6:],
        {"role": "user", "content": user_text}
    ]

    chat = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=0.5,
    )

    return chat.choices[0].message.content.strip()


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Defina OPENAI_API_KEY no arquivo .env")
    
    client = OpenAI()
    history = []

    print("Iniciando o assistente de voz. Pressione Ctrl+C para sair.")

    while True:
        try:
            audio = record_until_enter()
            if audio.size == 0:
                print("Nenhum áudio capturado. Tente novamente.")
                continue

            with temp_wav_path() as wav_path:
                sf.write(wav_path, audio, SAMPLERATE, subtype='PCM_16')
                user_text = transcribe_wav(client, wav_path).strip()

            if not user_text:
                print("Nenhum texto reconhecido. Tente novamente.")
                continue

            print(f"Você disse: {user_text}")

            if user_text.lower() in {"sair", "tchau", "até mais"}:
                print("Encerrando o assistente de voz. Até mais!")
                break

            assistant_text = chat_response(client, user_text, history)
            print(f"Assistente: {assistant_text}")

            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_text})
        except KeyboardInterrupt:
            print("\nEncerrando o assistente de voz. Até mais!")
            break

        except Exception as e:
            print(f"Erro: {e}. Tente novamente.")

if __name__ == "__main__":
    main()
    