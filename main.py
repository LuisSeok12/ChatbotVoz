import os
from dotenv import load_dotenv
from openai import OpenAI

from audio_utils import (
    record_until_enter, temp_wav_path, save_wav,
    play_wav_bytes, SAMPLE_RATE
)

from stt import transcribe_wav
from llm import chat_response
from tts import tts_wav_bytes

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Defina OPENAI_API_KEY no .env")

    client = OpenAI()
    history = []

    print("=== Voice Bot (refatorado: áudio + stt + llm + tts) ===")
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
