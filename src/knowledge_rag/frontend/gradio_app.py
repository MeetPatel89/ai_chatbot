import gradio as gr

from knowledge_rag.app.chat.chat import Chat
from knowledge_rag.config import load_dotenv
from knowledge_rag.frontend.gradio_logic import (
    PROVIDER_MODELS,
    clear_chat,
    on_provider_change,
    send,
)

load_dotenv()


with gr.Blocks(title="AI App UI") as demo:
    # One Chat object per session, registered as part of the app
    chat_obj = gr.State(Chat())
    gr.Markdown("## Minimal AI App UI (Gradio)")

    # --- Row 1: System Prompt (scale=1) | User Prompt (scale=3)
    with gr.Row():
        system_prompt = gr.Textbox(
            label="System Prompt",
            placeholder="e.g., 'You are a concise assistant...'",
            lines=5,
            scale=1,
            container=True,
        )
        user_prompt = gr.Textbox(
            label="User Prompt",
            placeholder="Type a message and press Enter or click Send",
            lines=5,
            scale=3,
            container=True,
        )

    # --- Row 3: Chatbox + controls
    with gr.Row():
        chat = gr.Chatbot(label="Chat History", type="messages", height=420, scale=3)
    with gr.Row():
        with gr.Column():
            provider = gr.Dropdown(
                label="Provider",
                choices=list(PROVIDER_MODELS.keys()),
                value="OpenAI",
            )
            model = gr.Dropdown(
                label="Model",
                choices=PROVIDER_MODELS["OpenAI"],
                value=PROVIDER_MODELS["OpenAI"][0],
            )
        with gr.Column():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Chat")

    # Interactions
    provider.change(fn=on_provider_change, inputs=provider, outputs=model)
    user_prompt.submit(
        fn=send,
        inputs=[user_prompt, system_prompt, provider, model, chat_obj],
        outputs=[chat, user_prompt],
    )
    send_btn.click(
        fn=send,
        inputs=[user_prompt, system_prompt, provider, model, chat_obj],
        outputs=[chat, user_prompt],
    )
    clear_btn.click(fn=clear_chat, inputs=[chat_obj], outputs=[chat, user_prompt])

if __name__ == "__main__":
    demo.launch()
