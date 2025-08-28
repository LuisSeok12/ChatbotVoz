# Voice Terminal Bot (PT-BR)

Chatbot de **terminal com voz** em Python que:

1. **Grava sua fala**,
2. **Transcreve** via OpenAI (STT),
3. **Gera resposta** (LLM),
4. **Fala de volta** (TTS).

Projeto organizado em **módulos** (áudio, STT, LLM, TTS) 

---

## Sumário

* [Requisitos](#requisitos)
* [Instalação](#instalação)
* [Configuração (.env)](#configuração-env)
* [Estrutura do Projeto](#estrutura-do-projeto)
* [Como Executar](#como-executar)
* [Uso](#uso)
* [Módulos](#módulos)

---

## Requisitos

* **Python 3.9+**
* Microfone funcional
* Conta e **API Key** da OpenAI

> Linux: `sounddevice` usa **PortAudio**. Se necessário, instale:
> Debian/Ubuntu: `sudo apt-get install portaudio19-dev`

---

## Instalação

### 1) Criar e ativar venv

**Windows (PowerShell):**

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS/Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Instalar dependências

```bash
pip install openai sounddevice soundfile numpy python-dotenv
```

> **Não** usamos `simpleaudio` (evita depender do MSVC no Windows).

---

## Configuração (.env)

Crie um arquivo **`.env`** na raiz:

```env
OPENAI_API_KEY=coloque_sua_chave_aqui

# Opcionais – têm defaults no código
# OPENAI_TEXT_MODEL=gpt-4o-mini
# OPENAI_STT_MODEL=gpt-4o-mini-transcribe
# OPENAI_TTS_MODEL=gpt-4o-mini-tts
# OPENAI_TTS_VOICE=alloy
```

---

## Estrutura do Projeto

```
voice_bot/
  .env
  main.py               # orquestra o fluxo fala→texto→resposta→fala
  audio_utils.py        # gravação, salvar WAV, tocar WAV
  stt.py                # transcrição (OpenAI)
  llm.py                # resposta (OpenAI)
  tts.py                # síntese de voz (OpenAI)
```

---

## Como Executar

```bash
python main.py
```

---

## Uso

* Fale após o prompt.
* O bot **transcreve**, **responde** e **fala** a resposta.
* Para encerrar, diga **“sair”** (ou `Ctrl+C`).

> O modo de captura pode estar configurado para:
>
> * **ENTER→fala→ENTER** (função `record_until_enter`), ou


---

## Módulos

### `audio_utils.py`

* `record_until_enter()` – grava até ENTER.
* `record_until_silence(...)` – grava até detectar **silêncio** (VAD por energia RMS).
* `save_wav(audio, path, SAMPLE_RATE)` – salva `numpy int16` em WAV PCM\_16.
* `play_wav_bytes(wav_bytes)` – toca WAV diretamente de bytes (memória).
* `temp_wav_path()` – context manager de arquivo temporário.

### `stt.py`

* `transcribe_wav(client, wav_path)` – envia WAV para **OpenAI STT** (ex.: `gpt-4o-mini-transcribe`) e retorna **texto**.

### `llm.py`

* `chat_response(client, user_text, history)` – usa **OpenAI** (ex.: `gpt-4o-mini`) para responder em **PT-BR**, mantendo **histórico curto**.

### `tts.py`

* `tts_wav_bytes(client, text)` – gera **WAV** da resposta via **OpenAI TTS** (ex.: `gpt-4o-mini-tts` + `alloy`).

### `main.py`

* Carrega `.env`, cria `OpenAI()`, controla o **loop**:

  1. grava (ENTER ou VAD) →
  2. **STT** →
  3. **LLM** →
  4. **TTS** (toca) →
  5. atualiza **histórico**.

---

