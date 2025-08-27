import os
from openai import OpenAI

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "Você é um assistente de voz objetivo e educado. Responda em PT-BR."
)

def chat_response(client: OpenAI, user_text: str, history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_text})

    chat = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=0.5,
    )
    return chat.choices[0].message.content.strip()
