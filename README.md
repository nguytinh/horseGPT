# HorseGPT 🐎💬
A local LLM-powered chatbot interface for exploring horse racing topics (or anything else!) using Llama 2 7B Chat, served locally via llama-cpp-python.
🧠 Requirements

Python 3.10+
Node.js + npm
Llama 2 GGUF model (e.g. llama-2-7b-chat.Q4_K_M.gguf)
llama-cpp-python installed with [server] extras

## 🚀 Installation & Setup

### Clone this repository

### Install web app dependencies
```
cd horsegpt_app
npm install
```

### Start the Backend Server
```
cd horsegpt_app
node server.js
```
### Launch the Web App
Open another terminal window:
```
cd horsegpt_app
npm start
```

Ensure you have the Llama 2 model downloaded to your preferred location

## 🏃‍♂️ Running the Application
First Open a new terminal window:

### Navigate to your model directory
```cd ~/llama-models```

### Activate your virtual environment (if using one)
```source llama-env/bin/activate```

### Start the Llama model server
```
python3 -m llama_cpp.server \
  --model ~/480/gemma-3-finetune.Q8_0_updated.gguf \
  --host 0.0.0.0 \
  --port 8080
```
The application should now be running and accessible in your web browser.

## 🔍 Features
Chat interface for horse racing topics\
Powered by Gemma3 4b Chat running locally\
Customizable prompts and settings

### Our current best model (Gemma3 4b)
https://drive.google.com/file/d/1pjyRJpZI7dH192t9n6eWQaPtvnt_6jkH/view?usp=sharing
