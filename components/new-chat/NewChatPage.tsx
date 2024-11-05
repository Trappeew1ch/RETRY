"use client";

import { useState, useRef, useEffect } from "react";
import { AnimatePresence } from "framer-motion";
import { InitialView } from "./InitialView";
import { ChatView } from "./ChatView";
import { Message } from "./types";
import { BackgroundEffects } from "./BackgroundEffects";

export default function NewChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isFirstMessage = messages.length === 0;

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim() || loading) return;

    setLoading(true);
    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      type: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");

    try {
      // Add thinking state
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        content: "Analyzing request...",
        type: 'assistant',
        timestamp: new Date(),
        status: 'thinking'
      }]);

      await new Promise(resolve => setTimeout(resolve, 1000));

      // Add generating state
      setMessages(prev => [...prev.filter(m => m.status !== 'thinking'), {
        id: Date.now().toString(),
        content: "Generating response...",
        type: 'assistant',
        timestamp: new Date(),
        status: 'generating'
      }]);

      await new Promise(resolve => setTimeout(resolve, 1000));

      // Simulate AI response based on input
      const response = generateResponse(input);

      // Final response
      setMessages(prev => [...prev.filter(m => !m.status), {
        id: Date.now().toString(),
        content: response.content,
        type: 'assistant',
        timestamp: new Date(),
        code: response.code
      }]);
    } catch (error) {
      console.error('Chat Error:', error);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        content: "I apologize, but I encountered an error. Please try again.",
        type: 'assistant',
        timestamp: new Date()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const generateResponse = (userInput: string) => {
    const input = userInput.toLowerCase();
    
    if (input.includes('hello') || input.includes('hi')) {
      return {
        content: "Hello! How can I assist you with your development needs today?",
        code: null
      };
    }
    
    if (input.includes('code') || input.includes('example')) {
      return {
        content: "Here's a simple example of a React component:",
        code: `import React from 'react';

const ExampleComponent = () => {
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h1 className="text-2xl font-bold">Hello World</h1>
      <p className="mt-2 text-gray-600">
        This is an example component
      </p>
    </div>
  );
};

export default ExampleComponent;`
      };
    }

    return {
      content: "I understand your request. How can I help you further with that?",
      code: null
    };
  };

  return (
    <div className="flex-1 relative overflow-hidden">
      <BackgroundEffects />
      
      <AnimatePresence mode="wait">
        {isFirstMessage ? (
          <InitialView
            input={input}
            setInput={setInput}
            loading={loading}
            handleSubmit={handleSubmit}
            fileInputRef={fileInputRef}
          />
        ) : (
          <ChatView
            messages={messages}
            input={input}
            setInput={setInput}
            loading={loading}
            handleSubmit={handleSubmit}
            chatContainerRef={chatContainerRef}
            fileInputRef={fileInputRef}
          />
        )}
      </AnimatePresence>
    </div>
  );
}