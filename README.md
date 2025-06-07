# HorseGPT 🐎💬
A local LLM-powered chatbot interface for exploring horse racing topics (or anything else!) using Llama 2 7B Chat, served locally via llama-cpp-python.
🧠 Requirements

Python 3.10+
Node.js + npm
Llama 2 GGUF model (e.g. llama-2-7b-chat.Q4_K_M.gguf)
llama-cpp-python installed with [server] extras

We modified this pipeline with an added additional cell for evaluation (see training.py) and did minor changes throughout: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb

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

## Our current best model (Gemma3 4b)
https://drive.google.com/file/d/1pjyRJpZI7dH192t9n6eWQaPtvnt_6jkH/view?usp=sharing

## Example Prompts
Predict the winning horse for the upcoming race.\nRace Details:\n Course: Lingfield (AW)\n Race Title: Ladbrokes Where The Nation Plays Fillies' Novice Stakes\n Date: 20/02/08\n Distance: N/A\n Condition: Standard\n Class: Class 5\n Ages: 3yo+\n Number of Runners: 7\n\nRunners:\n - Horse: Confounding, Age: 3, Jockey: Hayley Turner, Trainer: Charlie Fellowes, Weight: 8st 11lb, Odds (decimal): 0.067, RPR: 54, TR: 23, OR: N/A, Headgear: None\n - Horse: Silent Witness, Age: 4, Jockey: Luke Morris, Trainer: Ed Walker, Weight: 9st 12lb, Odds (decimal): 0.167, RPR: 61, TR: 33, OR: N/A, Headgear: None\n - Horse: New Arrival, Age: 3, Jockey: Ben Curtis, Trainer: James Tate, Weight: 8st 11lb, Odds (decimal): 0.308, RPR: 41, TR: 6, OR: 74, Headgear: None\n - Horse: Rare Glam, Age: 3, Jockey: Levi Williams, Trainer: Joseph Tuite, Weight: 8st 4lb, Odds (decimal): 0.020, RPR: 48, TR: 15, OR: N/A, Headgear: None\n - Horse: Cuckoo Clock, Age: 4, Jockey: Franny Norton, Trainer: Mark Johnston, Weight: 9st 12lb, Odds (decimal): 0.286, RPR: 64, TR: 37, OR: N/A, Headgear: None\n - Horse: Ahorsecalledwanda, Age: 3, Jockey: Joey Haynes, Trainer: Amanda Perrett, Weight: 8st 11lb, Odds (decimal): 0.308, RPR: 76, TR: 50, OR: N/A, Headgear: None\n - Horse: Daisy Green, Age: 3, Jockey: Thore Hammer Hansen, Trainer: Bill Turner, Weight: 8st 8lb, Odds (decimal): 0.024, RPR: 42, TR: 7, OR: N/A, Headgear: None

expected : Ahorsecalledwanda

Predict the winning horse for the upcoming race.\nRace Details:\n Course: Plumpton\n Race Title: Download The Free At The Races App Handicap Hurdle\n Date: 20/01/05\n Distance: N/A\n Condition: Soft\n Class: Class 5\n Ages: 4yo+\n Number of Runners: 15\n\nRunners:\n - Horse: Zoltan Varga, Age: 6, Jockey: Marc Goldstein, Trainer: Sheena West, Weight: 11st 12lb, Odds (decimal): 0.091, RPR: 101, TR: 93, OR: 100, Headgear: None\n - Horse: Cassivellaunus, Age: 8, Jockey: Niall Houlihan, Trainer: Daniel Steele, Weight: 10st 13lb, Odds (decimal): 0.020, RPR: N/A, TR: N/A, OR: 94, Headgear: None\n - Horse: Sauvignon, Age: 9, Jockey: Aidan Coleman, Trainer: George Baker, Weight: 11st 7lb, Odds (decimal): 0.048, RPR: 81, TR: 72, OR: 95, Headgear: None\n - Horse: Royal Concorde, Age: 9, Jockey: Brendan Powell, Trainer: Linda Jewell, Weight: 10st 4lb, Odds (decimal): 0.048, RPR: N/A, TR: N/A, OR: 78, Headgear: t\n - Horse: Kalarika, Age: 7, Jockey: Harry Cobden, Trainer: Colin Tizzard, Weight: 11st 12lb, Odds (decimal): 0.059, RPR: N/A, TR: N/A, OR: 100, Headgear: tb\n - Horse: Kalabee, Age: 5, Jockey: Tom Bellamy, Trainer: Keiran Burke, Weight: 10st 10lb, Odds (decimal): 0.020, RPR: 5, TR: N/A, OR: 84, Headgear: tp\n - Horse: High Up In The Air, Age: 6, Jockey: Joshua Moore, Trainer: Gary Moore, Weight: 11st 5lb, Odds (decimal): 0.182, RPR: 108, TR: 95, OR: 93, Headgear: None\n - Horse: Ede'iffs Elton, Age: 6, Jockey: Rex Dingle, Trainer: Robert Walford, Weight: 11st 6lb, Odds (decimal): 0.083, RPR: 99, TR: 90, OR: 97, Headgear: None\n - Horse: Bostin, Age: 12, Jockey: Thomas Garner, Trainer: Daniel O'Brien, Weight: 10st 8lb, Odds (decimal): 0.010, RPR: 24, TR: 13, OR: 82, Headgear: None\n - Horse: Schap, Age: 8, Jockey: Bridget Andrews, Trainer: Caroline Fryer, Weight: 11st 3lb, Odds (decimal): 0.010, RPR: N/A, TR: N/A, OR: 91, Headgear: p\n - Horse: Affaire D'Honneur, Age: 9, Jockey: Harry Bannister, Trainer: Tony Carroll, Weight: 11st 4lb, Odds (decimal): 0.125, RPR: 56, TR: 47, OR: 92, Headgear: t\n - Horse: Nessfield Blue, Age: 6, Jockey: Mark Grant, Trainer: Pat Murphy, Weight: 11st 12lb, Odds (decimal): 0.059, RPR: 63, TR: 54, OR: 100, Headgear: None\n - Horse: Vicenzo Mio, Age: 10, Jockey: Leighton Aspell, Trainer: Chris Gordon, Weight: 11st 9lb, Odds (decimal): 0.059, RPR: 67, TR: 58, OR: 97, Headgear: tp\n - Horse: Thats My Rabbit, Age: 11, Jockey: Richard Johnson, Trainer: Suzi Best, Weight: 11st 8lb, Odds (decimal): 0.231, RPR: 83, TR: 70, OR: 96, Headgear: None\n - Horse: The Premier Celtic, Age: 7, Jockey: Sean Houlihan, Trainer: Pat Phelan, Weight: 11st 8lb, Odds (decimal): 0.143, RPR: 85, TR: 74, OR: 99, Headgear: p

expected : High Up In The Air

To generate more, see the prompt generation code in our paper (or format.py in this github) & use the dataset listed in our paper.


