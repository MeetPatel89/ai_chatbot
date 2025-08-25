# gradio_logic.py
import gradio as gr

from knowledge_rag.app.providers import get_llm_client

PROVIDER_MODELS = {
    "OpenAI": [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4.1-mini",
    ],
    "Anthropic": [
        "claude-3-5-haiku-latest",
        "claude-3-7-sonnet-latest",
        "claude-sonnet-4-0",
    ],
    "Grok": ["grok-2", "grok-2-mini"],
    "Google": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "Local": ["llama-3.1-8b", "mistral-nemo", "phi-3.5-mini"],
}


def on_provider_change(provider: str):
    print(f"Provider changed to: {provider}")
    models = PROVIDER_MODELS.get(provider, [])
    print(f"Available models: {models}")
    default = models[0] if models else None
    print(f"Default model: {default}")
    return gr.update(choices=models, value=default)


def send(user_msg, system_msg, provider, model, chat_obj):
    llm_client = get_llm_client(provider, model)
    chat_obj.set_llm_client(llm_client)
    chat_obj.set_system_message(system_msg)
    chat_obj.send_message({"role": "user", "content": user_msg})
    return chat_obj.get_chat_history(), ""  # clear the user prompt


def clear_chat(chat_obj):
    chat_obj.clear_chat_history()
    return chat_obj.get_chat_history(), ""
