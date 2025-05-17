import React, { useState, useRef, useEffect } from 'react';
import { Spinner } from '@patternfly/react-core';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8080/v1/completions', {
        model: "./llama-2-7b-chat.Q4_K_M.gguf",
        prompt: `<s>[INST] ${input} [/INST]`,
        max_tokens: 2000,
        temperature: 0.7
      });

      const reply = response.data.choices[0].text.trim();
      setMessages(prev => [...prev, { role: 'assistant', content: reply }]);
    } catch (error) {
      console.error('Error calling Llama API:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, there was an error connecting to the Llama model. Make sure the server is running.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>HorseGPT</h1>
      </header>
      
      <main className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-container">
            <h2>Welcome to HorseGPT</h2>
            <p>Ask me anything about Horse Racing to get started!</p>
            <div className="horse-icon">🐎</div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${message.role === 'user' ? 'user-message' : 'bot-message'}`}
              >
                <div className="message-avatar">
                  {message.role === 'user' ? '👤' : '🐎'}
                </div>
                <div className="message-bubble">
                  {message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message bot-message">
                <div className="message-avatar">🐎</div>
                <div className="message-bubble loading-bubble">
                  <Spinner size="md" /> Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>
      
      <footer className="chat-input-area">
        <div className="input-container">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            placeholder="Type your message here..."
            rows={1}
          />
          <button
            className="send-button"
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
          >
            →
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;