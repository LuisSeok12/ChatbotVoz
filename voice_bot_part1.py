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

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
STT_MODEL  = os.getenv("OPENAI_STT_MODEL",  "gpt-4o-mini-transcribe")
TTS_MODEL  = os.getenv("OPENAI_TTS_MODEL",  "gpt-4o-mini-tts")
TTS_VOICE  = os.getenv("OPENAI_TTS_VOICE",  "alloy")

# =========================
# ÁUDIO: gravar e tocar
# =========================
def record_until_enter() -> np.ndarray:
    input("\n[ENTER] para INICIAR a gravação...")
    print("Gravando... [ENTER] para PARAR")

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            pass
        q.put(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=callback)
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
        return np.zeros((0, CHANNELS), dtype=DTYPE)
    audio = np.concatenate(chunks, axis=0)
    print(f"Capturado: {audio.shape[0]/SAMPLE_RATE:.2f}s de áudio.")
    return audio

def play_wav_bytes(wav_bytes: bytes):
    # Toca WAV diretamente da memória
    import io
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(data, sr)
    sd.wait()

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

# =========================
# OpenAI: STT, LLM, TTS
# =========================
def transcribe_wav(client: OpenAI, wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        text = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            response_format="text",
        )
    return text if isinstance(text, str) else getattr(text, "text", str(text))

def chat_response(client: OpenAI, user_text: str, history: list) -> str:
    messages = [
        {"role": "system", "content": "Você é um assistente de voz objetivo e educado. Responda em PT-BR."},
        *history[-6:],
        {"role": "user", "content": user_text},
    ]
    chat = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=0.5,
    )
    return chat.choices[0].message.content.strip()

def tts_wav_bytes(client: OpenAI, text: str) -> bytes:
    # Retorna bytes WAV da fala gerada
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
    )
    # SDK recente: dá para salvar com to_file(); aqui voltamos bytes
    return speech.read() if hasattr(speech, "read") else bytes(speech)

# =========================
# MAIN LOOP (fala ↔ texto ↔ fala)
# =========================
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Defina OPENAI_API_KEY no arquivo .env")

    client = OpenAI()
    history = []

    print("=== Voice Bot (Parte 2: STT + LLM + TTS) ===")
    print("Diga 'sair' para encerrar.\n")

    while True:
        try:
            # 1) grava
            audio = record_until_enter()
            if audio.size == 0:
                print("Nenhum áudio capturado. Tente novamente.")
                continue

            # 2) salva temp e transcreve
            with temp_wav_path() as wav_path:
                sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
                user_text = transcribe_wav(client, wav_path).strip()

            if not user_text:
                print("Transcrição vazia. Tente novamente.")
                continue

            print(f"\nVocê: {user_text}")
            if user_text.lower() in {"sair", "exit", "quit"}:
                print("Encerrando...")
                break

            # 3) responde (texto)
            assistant_text = chat_response(client, user_text, history)
            print(f"Assistente: {assistant_text}\n")

            # 4) fala (TTS)
            try:
                wav_bytes = tts_wav_bytes(client, assistant_text)
                play_wav_bytes(wav_bytes)
            except Exception as e:
                print(f"[Aviso TTS] Não foi possível sintetizar/tocar: {e}")

            # 5) histórico
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_text})

        except KeyboardInterrupt:
            print("\nInterrompido. Até mais!")
            break
        except Exception as e:
            print(f"[ERRO] {e}")

if __name__ == "__main__":
    main()
