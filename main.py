import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from audio_utils import (
    record_until_enter,
    temp_wav_path,
    save_wav,
    play_wav_bytes,
    SAMPLE_RATE,
)

# =========================
# MODELOS / CONFIG
# =========================
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
STT_MODEL  = os.getenv("OPENAI_STT_MODEL",  "gpt-4o-mini-transcribe")
TTS_MODEL  = os.getenv("OPENAI_TTS_MODEL",  "gpt-4o-mini-tts")
TTS_VOICE  = os.getenv("OPENAI_TTS_VOICE",  "alloy")


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
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
    )
    return speech.read() if hasattr(speech, "read") else bytes(speech)


# =========================
# MAIN LOOP
# =========================
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Defina OPENAI_API_KEY no .env")

    client = OpenAI()
    history = []

    print("=== Voice Bot (refatoração etapa 1: módulo de áudio) ===")
    print("Diga 'sair' para encerrar.\n")

    while True:
        try:
            # 1) Gravar
            audio = record_until_enter()
            if audio.size == 0:
                print("Nenhum áudio capturado. Tente novamente.")
                continue

            # 2) Salvar & Transcrever
            with temp_wav_path() as wav_path:
                save_wav(audio, wav_path, SAMPLE_RATE)
                user_text = transcribe_wav(client, wav_path).strip()

            if not user_text:
                print("Transcrição vazia. Tente novamente.")
                continue

            print(f"\nVocê: {user_text}")
            if user_text.lower() in {"sair", "exit", "quit"}:
                print("Encerrando...")
                break

            # 3) LLM
            assistant_text = chat_response(client, user_text, history)
            print(f"Assistente: {assistant_text}\n")

            # 4) TTS
            try:
                wav_bytes = tts_wav_bytes(client, assistant_text)
                play_wav_bytes(wav_bytes)
            except Exception as e:
                print(f"[Aviso TTS] Não foi possível sintetizar/tocar: {e}")

            # 5) Histórico
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_text})

        except KeyboardInterrupt:
            print("\nInterrompido. Até mais!")
            break
        except Exception as e:
            print(f"[ERRO] {e}")

if __name__ == "__main__":
    main()
