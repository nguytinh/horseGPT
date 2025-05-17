// server.js
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const app = express();

app.use(cors());
app.use(express.json());

const PORT = 3001;
const LLAMA_API_URL = 'http://localhost:8080/v1/completions';

app.post('/api/chat', async (req, res) => {
  try {
    const { prompt } = req.body;

    const response = await axios.post(LLAMA_API_URL, {
      model: "./llama-2-7b-chat.Q4_K_M.gguf",
      prompt: `<s>[INST] ${prompt} [/INST]`,
      max_tokens: 2000,
      temperature: 0.7
    });

    console.log('Raw LLaMA API Response:', response.data);

    const reply = response.data.choices[0].text.trim();
    res.json({ response: reply });

  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).json({ error: 'Failed to get response from Llama' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
