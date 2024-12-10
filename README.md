# Ollama Code Artefacts

## Running the code

1. Create a venv and install the requirements
2. Copy the `.env.example` file to `.env` and adjust the configuration as desired
3. Run the server

```bash
uv venv
source .venv/bin/activate
uv pip install -U -r requirements.txt
cp .env.example .env
python app.py
```

## Configuration

The configuration is done through the `.env` file. The following variables are available:

- `OLLAMA_MODEL` default: `qwen2.5-coder:14b-instruct-q6_K` - Model to use
- `OLLAMA_HOST` default: `http://localhost:11434` - Host URL of the Ollama server
- `MAX_TOKENS` default: `4096` - Maximum number of tokens to generate
- `OPEN_BROWSER` default: `True` - Open the browser when the server starts

## Screenshot

![screenshot](static/screenshot.png)

---

Originally forked from https://huggingface.co/spaces/Qwen/Qwen2.5-Coder-Artifacts
