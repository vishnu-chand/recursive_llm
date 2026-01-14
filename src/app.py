import os
import logging
import traceback
import gradio as gr
from rlm import RecursiveLanguageModel, get_logger

logger = get_logger("APP")
logger.setLevel(logging.INFO)
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 980
RLM_MAX_STEPS = 15
TEXTBOX_HEIGHT = 10
RLM_MAX_CHUNK_SIZE = 10000
RLM_MAX_OUTPUT_CHARS = 10000
SIDEBAR_WIDTH = int(SCREEN_WIDTH * 0.15)
CHATBOT_HEIGHT = int(SCREEN_HEIGHT * 0.9)


def load_context_from_file(file_obj):
    if file_obj is None:
        return None, gr.Info("No file selected.")
    try:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Loaded context from file. Size: {len(content)} chars.")
        return content, gr.Info(f"File loaded successfully! ({len(content)} chars)")
    except Exception as e:
        logger.error(f"Error reading file: {traceback.format_exc()}")
        return None, gr.Error(f"Failed to read file: {e}")


def generate_bot_response(history, context_text, model_name):
    """Processes the LLM response after the user message is visible."""
    if history:
        response = "⚠️ **No Context Loaded.**\nPlease upload a file or paste text in the sidebar to define the 'CONTEXT' environment."
        if context_text:
            try:
                logger.info(f"🤖 Processing {model_name} response...")
                rlm = RecursiveLanguageModel(model_name=model_name)
                response = rlm.run(context_text, history, max_chunk_size=RLM_MAX_CHUNK_SIZE, max_output_chars=RLM_MAX_OUTPUT_CHARS, max_steps=RLM_MAX_STEPS)
            except Exception:
                response = f"❌ **Error during execution:**\n\n{traceback.format_exc()}"
                logger.error(response)
        history.append({"role": "assistant", "content": str(response)})
    return history


def create_chat_interface():
    gr.Markdown("### 🤖 RLM Chat Agent")
    chatbot = gr.Chatbot(height=CHATBOT_HEIGHT, placeholder="Upload a context file on the left, then ask questions here.")

    with gr.Row():
        msg_input = gr.Textbox(show_label=False, placeholder="Type your query...", scale=5, container=False, autofocus=True)
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", variant="stop", scale=1)
    return chatbot, clear_btn, msg_input, send_btn


def create_sidebar_components():
    gr.Markdown("### ⚙️ Settings")
    model_input = gr.Dropdown(
        choices=[
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.5-pro",
            "gemini/gemini-3-pro-preview",
            "anthropic/claude-3-opus-20240229",
            "openai/gpt-4-turbo-preview",
            "openai/gpt-4",
        ],
        value="gemini/gemini-2.5-flash",
        allow_custom_value=True,
        label="Model Name",
        info="Select or type model name.",
    )
    gr.Markdown("### 📚 Context")
    gr.Markdown("Upload a document or paste text.")

    with gr.Tab("Upload File"):
        file_input = gr.File(label="Context File", file_types=[".txt", ".md"], interactive=True)

    with gr.Tab("Paste Text"):
        text_input = gr.Textbox(label="Manual Context", lines=TEXTBOX_HEIGHT, placeholder="Paste long text here...")
        paste_btn = gr.Button("Load Pasted Context", variant="secondary")

    context_status = gr.Markdown("🔴 *No context loaded*")
    return context_status, file_input, model_input, paste_btn, text_input


def create_ui():
    def add_user_message(message, history):
        if message:
            history.append({"role": "user", "content": str(message)})
        return "", history

    def handle_file_upload_wrapper(file_obj, context_state):
        if file_obj is not None:
            content, status = load_context_from_file(file_obj)
            if content:
                context_state = content
                return context_state, gr.Markdown("🟢 *Context loaded from file*")
        return "", gr.Markdown("🔴 *No context loaded*")

    def handle_text_paste_wrapper(data, context_state):
        if data.strip():
            context_state = data
            return context_state, gr.Markdown("🟢 *Context loaded from text*")
        return context_state, gr.Markdown("🔴 *No context loaded*")

    with gr.Blocks(title="Recursive Language Model Agent", theme=gr.themes.Soft()) as demo:
        context_state = gr.State(value="")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=SIDEBAR_WIDTH, variant="panel"):
                context_status, file_input, model_input, paste_btn, text_input = create_sidebar_components()
            with gr.Column(scale=4):
                chatbot, clear_btn, msg_input, send_btn = create_chat_interface()

        file_input.upload(fn=handle_file_upload_wrapper, inputs=[file_input, context_state], outputs=[context_state, context_status])
        paste_btn.click(fn=handle_text_paste_wrapper, inputs=[text_input, context_state], outputs=[context_state, context_status])
        msg_input.submit(fn=add_user_message, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False).then(fn=generate_bot_response, inputs=[chatbot, context_state, model_input], outputs=chatbot)
        send_btn.click(fn=add_user_message, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False).then(fn=generate_bot_response, inputs=[chatbot, context_state, model_input], outputs=chatbot)
        clear_btn.click(lambda: [], None, chatbot, queue=False)

    return demo


def main():
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
