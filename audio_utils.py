import os
import time
import queue
import tempfile
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import sounddevice as sd
import soundfile as sf


# =========================
# CONFIG PADRÃO
# =========================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


# =========================
# GRAVAÇÃO
# =========================
def record_until_enter(
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    dtype: str = DTYPE,
) -> np.ndarray:
    """
    Pressione ENTER para iniciar e ENTER para parar.
    Retorna áudio mono int16.
    """
    input("\n[ENTER] para INICIAR a gravação...")
    print("Gravando... [ENTER] para PARAR")

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            # você pode logar o status se quiser
            pass
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype=dtype,
        callback=callback,
    )
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
        return np.zeros((0, channels), dtype=dtype)

    audio = np.concatenate(chunks, axis=0)
    print(f"Capturado: {audio.shape[0] / sample_rate:.2f}s de áudio.")
    return audio


# =========================
# TOCAR ÁUDIO (WAV em bytes)
# =========================
def play_wav_bytes(wav_bytes: bytes) -> None:
    """Toca áudio WAV diretamente de bytes (memória)."""
    import io
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(data, sr)
    sd.wait()


# =========================
# ARQUIVOS TEMPORÁRIOS
# =========================
@contextmanager
def temp_wav_path(prefix: str = "user_", suffix: str = ".wav") -> Iterator[str]:
    """
    Gera caminho temporário para WAV e apaga ao sair do contexto.
    """
    path = os.path.join(tempfile.gettempdir(), f"{prefix}{int(time.time() * 1000)}{suffix}")
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def save_wav(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Salva array numpy int16 como WAV PCM_16."""
    sf.write(path, audio, sample_rate, subtype="PCM_16")
