import { ChatProvider } from './components/ChatUI'
import ChatUI from './components/ChatUI'
import './index.css'

export default function App() {
  return (
      <ChatProvider>
      <ChatUI />
    </ChatProvider>
  );
}
