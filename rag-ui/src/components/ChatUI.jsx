import React, { useState, useEffect, useRef, createContext, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Cpu } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './ChatUI.css';

const ThemeContext = createContext({ theme: 'light', toggleTheme: () => {} });

const conversations = [
  { id: 1, title: 'New Chat' },
];

const renderMessage = (msg, index) => (
  <motion.div
    key={index}
    className="message-wrapper"
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0 }}
    transition={{ duration: 0.3 }}
  >
    <div className="message-icon">
      {msg.from === 'user' ? <User size={24} /> : <Cpu size={24} />}
    </div>
    <div className={`message ${msg.from}`}>
      {msg.from === 'user' ? (
        <p>{msg.text}</p>
      ) : (
        <div className="markdown-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {msg.text || ''}
          </ReactMarkdown>
        </div>
      )}
      
      {msg.from === 'ai' && msg.sources?.length > 0 && (
        <details className="sources">
          <summary>Sources ({msg.sources.length})</summary>
          <div className="sources-list">
            {msg.sources.map((source, idx) => (
              <div key={idx} className="source-item">
                <h4>{source.title || 'Untitled'}</h4>
                <p>{source.excerpt || 'No excerpt available'}</p>
                <span className="relevance">{source.relevance || 'N/A'}</span>
              </div>
            ))}
          </div>
        </details>
      )}

      {msg.from === 'ai' && msg.metrics && (
        <div className="metrics">
          <small>
            Response time: {msg.metrics?.query_time?.toFixed(2) || '0.00'}s | 
            Sources: {msg.metrics?.num_results || 0} | 
            Relevance: {(msg.metrics?.avg_similarity * 100 || 0).toFixed(1)}%
          </small>
        </div>
      )}
      
      <span className="time">
        {msg.time?.toLocaleTimeString() || new Date().toLocaleTimeString()}
      </span>
    </div>
  </motion.div>
);

export function ChatProvider({ children }) {
  const [theme, setTheme] = useState('light');
  
  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  useEffect(() => {
    document.documentElement.className = theme;
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export default function ChatUI() {
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [activeConv, setActiveConv] = useState(conversations[0].id);
  const { theme, toggleTheme } = useContext(ThemeContext);
  const scrollRef = useRef();

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    setIsLoading(true);
    const userMsg = { text: input.trim(), from: 'user', time: new Date() };
    const queryText = input.trim();
    setInput('');
    setMessages(prev => [...prev, userMsg]);

    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: queryText,
          top_k: 3
        })
      });
      
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      
      const data = await res.json();
      
      const aiMsg = {
        text: data.answer || 'No response received',
        from: 'ai',
        time: new Date(),
        sources: data.sources || [],
        metrics: {
          query_time: data.metrics?.query_time || 0,
          num_results: data.metrics?.num_results || 0,
          avg_similarity: data.metrics?.avg_similarity || 0
        }
      };
      
      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      console.error('API Error:', error);
      const errMsg = {
        text: `Error: ${error.message || 'Unknown error occurred'}`,
        from: 'ai',
        time: new Date()
      };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app-container">
      <aside>
        <h2>Conversations</h2>
        <ul>
          {conversations.map(conv => (
            <li key={conv.id}>
              <button
                className={`conv-btn ${activeConv === conv.id ? 'active' : ''}`}
                onClick={() => setActiveConv(conv.id)}
              >
                {conv.title}
              </button>
            </li>
          ))}
        </ul>
        <button className="toggle-btn" onClick={toggleTheme}>
          Toggle {theme === 'dark' ? 'Light' : 'Dark'}
        </button>
      </aside>

      <div className="chat-area">
        <header>
          <h3>{conversations.find(c => c.id === activeConv)?.title || 'New Chat'}</h3>
          <button className="settings-btn">Settings</button>
        </header>

        <main ref={scrollRef}>
          <AnimatePresence initial={false}>
            {messages.map((msg, i) => renderMessage(msg, i))}
          </AnimatePresence>
          {isLoading && (
            <div className="message-wrapper loading">
              <div className="message-icon">
                <Cpu size={24} />
              </div>
              <div className="message ai">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
        </main>

        <footer>
          <input
            type="text"
            placeholder="Ask about Dublin's planning..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            disabled={isLoading}
          />
          <button 
            className="send-btn" 
            onClick={sendMessage}
            disabled={isLoading}
          >
            Send
          </button>
        </footer>
      </div>
    </div>
  );
}