import React from 'react';
import ReactDOM from 'react-dom/client';
import ChatUI, { ChatProvider } from './components/ChatUI';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ChatProvider>
      <ChatUI />
    </ChatProvider>
  </React.StrictMode>
);