HorseGPT 🐎💬
A local LLM-powered chatbot interface for exploring horse racing topics (or anything else!) using Llama 2 7B Chat, served locally via llama-cpp-python.
🧠 Requirements

Python 3.10+
Node.js + npm
Llama 2 GGUF model (e.g. llama-2-7b-chat.Q4_K_M.gguf)
llama-cpp-python installed with [server] extras

🚀 Installation & Setup

Clone this repository
bashgit clone https://github.com/yourusername/horsegpt.git
cd horsegpt

Install web app dependencies
bashcd horsegpt_app
npm install

Ensure you have the Llama 2 model downloaded to your preferred location

🏃‍♂️ Running the Application
Step 1: Start the Llama Model Server
bash# Navigate to your model directory
cd ~/llama-models

# Activate your virtual environment (if using one)
source llama-env/bin/activate

# Start the Llama model server
python3 -m llama_cpp.server \
  --model ./llama-2-7b-chat.Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080
Step 2: Start the Backend Server
Open a new terminal window:
bashcd horsegpt_app
node server.js
Step 3: Launch the Web App
Open another terminal window:
bashcd horsegpt_app
npm start
The application should now be running and accessible in your web browser.
🔍 Features

Chat interface for horse racing topics
Powered by Llama 2 7B Chat running locally
Customizable prompts and settings
