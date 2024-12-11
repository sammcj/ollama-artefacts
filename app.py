import os
import re
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple
import base64
import aiohttp
import dotenv

# Import litellm and async_generator for Ollama streaming
import litellm
from async_generator import asynccontextmanager
import gradio as gr

import modelscope_studio.components.base as ms
import modelscope_studio.components.legacy as legacy
import modelscope_studio.components.antd as antd
from config import DEMO_LIST, SystemPrompt

# load dotenv
dotenv.load_dotenv()

# Configuration for Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b-instruct-q6_K")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
OPEN_BROWSER = os.getenv("OPEN_BROWSER", "False").lower() == "true"


History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]


async def get_ollama_models():
    """Fetch available models from Ollama server."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_HOST}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract model names from the response
                    models = [model["name"] for model in data["models"]]
                    return models
                return []
    except Exception as e:
        print(f"Error fetching Ollama models: {str(e)}")
        return []


def get_image_html():
    gif_path = os.path.join(os.path.dirname(__file__), "static/qwencode.gif")
    # Read the gif file as binary
    with open(gif_path, "rb") as f:
        gif_data = f.read()
    import base64

    # Convert to base64
    gif_base64 = base64.b64encode(gif_data).decode("utf-8")
    return f"""
    <div class="left_header">
     <img src="data:image/gif;base64,{gif_base64}" width="200px" />
     <h1>Code Generation with Ollama</h2>
    </div>
     """


def history_to_messages(history: History, system: str) -> Messages:
    messages = [{"role": "system", "content": system}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    return messages


def messages_to_history(messages: Messages) -> History:
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([q["content"], r["content"]])
    return history


def remove_code_block(text):
    pattern = r"```html\n(.+?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()


def history_render(history: History):
    return gr.update(open=True), history


def clear_history():
    return []


def send_to_sandbox(code):
    encoded_html = base64.b64encode(code.encode("utf-8")).decode("utf-8")
    data_uri = f"data:text/html;charset=utf-8;base64,{encoded_html}"
    return f'<iframe src="{data_uri}" width="100%" height="960px"></iframe>'


def demo_card_click(e: gr.EventData):
    index = e._data["component"]["index"]
    return DEMO_LIST[index]["description"]


async def update_model_list():
    """Update the model list dropdown."""
    models = await get_ollama_models()
    options = [{"label": model, "value": model} for model in models]
    return gr.update(options=options, value=OLLAMA_MODEL)


def process_html_content(content: str) -> str:
    """Safely process HTML content without trying to validate image paths"""
    try:
        # Remove code block markers if present
        content = remove_code_block(content)
        return content
    except Exception as e:
        print(f"Error processing HTML: {str(e)}")
        return content


async def generation_code(
    query: Optional[str], _setting: Dict[str, str], _history: Optional[History]
):
    if query is None:
        query = ""
    if _history is None:
        _history = []

    messages = history_to_messages(_history, _setting["system"])
    messages.append({"role": "user", "content": query})

    try:
        response = await litellm.acompletion(
            model=f"ollama/{_setting.get('model', OLLAMA_MODEL)}",
            messages=messages,
            api_base=OLLAMA_HOST,
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        content_buffer = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                content_buffer += chunk.choices[0].delta.content
                # Skip sandbox processing during streaming
                yield (
                    content_buffer,
                    _history,
                    None,
                    gr.update(active_key="loading"),
                    gr.update(open=True),
                )

        # Process final content
        processed_content = process_html_content(content_buffer)
        sandbox_content = (
            send_to_sandbox(processed_content) if processed_content else None
        )

        _history = messages_to_history(
            messages + [{"role": "assistant", "content": content_buffer}]
        )

        yield (
            content_buffer,
            _history,
            sandbox_content,
            gr.update(active_key="render"),
            gr.update(open=False),
        )

    except Exception as e:
        print(f"Error generating code: {str(e)}")
        yield (
            f"Error: {str(e)}",
            _history,
            None,
            gr.update(active_key="empty"),
            gr.update(open=True),
        )


with gr.Blocks(css_paths="app.css") as demo:
    history = gr.State([])
    setting = gr.State(
        {
            "system": SystemPrompt,
            "model": OLLAMA_MODEL,
        }
    )

    with ms.Application() as app:
        with antd.ConfigProvider():
            with antd.Row() as layout:

                with antd.Col(span=6):
                    with antd.Flex(
                        vertical=True,
                        gap="middle",
                        wrap=True,
                        elem_classes="middle-panel",
                    ):
                        header = gr.HTML(get_image_html())

                        input = antd.InputTextarea(
                            size="large",
                            allow_clear=True,
                            placeholder="Please enter what kind of application you want",
                        )
                        btn = antd.Button("Code It So", type="primary", size="large")
                        clear_btn = antd.Button(
                            "Clear History", type="default", size="large"
                        )

                        antd.Divider("Settings")

                        with antd.Flex(gap="small", wrap=True, align="center"):
                            settingPromptBtn = antd.Button(
                                "⚙️ System Prompt",
                                type="default",
                            )
                            historyBtn = antd.Button("📜 History", type="default")
                            modelBtn = antd.Button("🤖 Select Model", type="default")

                        antd.Divider("Examples")
                        with antd.Flex(gap="small", wrap=True):
                            with ms.Each(DEMO_LIST):
                                with antd.Card(
                                    hoverable=True, as_item="card"
                                ) as demoCard:
                                    antd.CardMeta()
                                demoCard.click(demo_card_click, outputs=[input])

                    # System prompt modal
                    with antd.Modal(
                        open=False, title="System Prompt", width="800px"
                    ) as system_prompt_modal:
                        systemPromptInput = antd.InputTextarea(
                            SystemPrompt, auto_size=True
                        )

                    # Model selection modal
                    with antd.Modal(
                        open=False, title="Select Model", width="400px"
                    ) as model_modal:
                        modelSelect = antd.Select(
                            options=[{"label": OLLAMA_MODEL, "value": OLLAMA_MODEL}],
                            value=OLLAMA_MODEL,
                            placeholder="Select Model",
                        )

                    # History drawer
                    with antd.Drawer(
                        open=False, title="history", placement="left", width="900px"
                    ) as history_drawer:
                        history_output = legacy.Chatbot(
                            show_label=False,
                            flushing=False,
                            height=960,
                            elem_classes="history_chatbot",
                        )

                    # Button handlers
                    settingPromptBtn.click(
                        lambda: gr.update(open=True),
                        inputs=[],
                        outputs=[system_prompt_modal],
                    )
                    system_prompt_modal.ok(
                        lambda input: ({"system": input}, gr.update(open=False)),
                        inputs=[systemPromptInput],
                        outputs=[setting, system_prompt_modal],
                    )
                    system_prompt_modal.cancel(
                        lambda: gr.update(open=False), outputs=[system_prompt_modal]
                    )

                    # Model selection handlers
                    async def handle_model_button():
                        models = await get_ollama_models()
                        options = [{"label": model, "value": model} for model in models]
                        return [
                            gr.update(open=True),
                            gr.update(options=options, value=OLLAMA_MODEL),
                        ]

                    modelBtn.click(
                        fn=handle_model_button,
                        outputs=[model_modal, modelSelect],
                    )

                    def handle_model_selection(current_setting, selected_model):
                        return {**current_setting, "model": selected_model}, gr.update(
                            open=False
                        )

                    model_modal.ok(
                        fn=handle_model_selection,
                        inputs=[setting, modelSelect],
                        outputs=[setting, model_modal],
                    )

                    model_modal.cancel(
                        fn=lambda: gr.update(open=False),
                        outputs=[model_modal],
                    )

                    historyBtn.click(
                        history_render,
                        inputs=[history],
                        outputs=[history_drawer, history_output],
                    )
                    history_drawer.close(
                        lambda: gr.update(open=False),
                        inputs=[],
                        outputs=[history_drawer],
                    )
                with antd.Col(span=6):
                    with ms.Div(elem_classes="code-panel"):
                        gr.HTML('<div class="panel-header">Generated Code</div>')
                        code_output = legacy.Markdown()

                with antd.Col(span=12):
                    with ms.Div(elem_classes="right_panel"):
                        gr.HTML(
                            '<div class="render_header"><span class="header_btn"></span><span class="header_btn"></span><span class="header_btn"></span></div>'
                        )
                        with antd.Tabs(
                            active_key="empty", render_tab_bar="() => null"
                        ) as state_tab:
                            with antd.Tabs.Item(key="empty"):
                                empty = antd.Empty(
                                    description="empty input",
                                    elem_classes="right_content",
                                )
                            with antd.Tabs.Item(key="loading"):
                                loading = antd.Spin(
                                    True,
                                    tip="coding...",
                                    size="large",
                                    elem_classes="right_content",
                                )
                            with antd.Tabs.Item(key="render"):
                                sandbox = gr.HTML(elem_classes="html_content")

            btn.click(
                fn=generation_code,
                inputs=[input, setting, history],
                outputs=[code_output, history, sandbox, state_tab],
            )

            clear_btn.click(clear_history, inputs=[], outputs=[history])

if __name__ == "__main__":

    # if running on macOS server_name=localhost, otherwise 0.0.0.0
    if os.uname().sysname == "Darwin":
        server_name = "localhost"
    else:
        server_name = "0.0.0.0"

    demo.queue(default_concurrency_limit=20).launch(
        server_name=server_name,
        server_port=7860,
        ssr_mode=False,
        share=False,
        inbrowser=OPEN_BROWSER,
    )
