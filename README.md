# HorseGPT 🐎💬

HorseGPT is a local LLM-powered chatbot interface for exploring horse racing topics (or anything else!) using Llama 2 7B Chat, served locally via `llama-cpp-python`.

---

## 🧠 Requirements

- Python 3.10+
- Node.js + npm
- Llama 2 GGUF model (e.g. `llama-2-7b-chat.Q4_K_M.gguf`)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) installed with `[server]` extras

---

## 🦙 Start the Llama Model Server

```bash
# 1. Go to your model directory
cd ~/llama-models

# 2. Activate your virtual environment
source llama-env/bin/activate

# 3. Start the Llama model server
python3 -m llama_cpp.server \
  --model ./llama-2-7b-chat.Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080

## **Run the Server in terminal**
# 1. cd horsegpt_app
# 2. npm i
# 3. node server.js

## **Start the Web App in another terminal**
# 1. npm start

