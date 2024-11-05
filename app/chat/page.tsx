"use client";

import { useState, useRef, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Navigation } from "@/components/chat/Navigation";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { ChatPreview } from "@/components/chat/ChatPreview";
import { Message, models, modes } from "@/types/chat";
import { sendChatMessage, processFileStructure } from "@/lib/api";

export default function ChatPage() {
  const [loading, setLoading] = useState(false);
  const [input, setInput] = useState("");
  const [selectedModel, setSelectedModel] = useState<typeof models[number]>("gemini-1.5-pro-002");
  const [selectedMode, setSelectedMode] = useState<typeof modes[number]['value']>("basic");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isNavExpanded, setIsNavExpanded] = useState(true);
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      type: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await sendChatMessage(input);

      // Process file structure if present
      if (response.files && response.files.length > 0) {
        await processFileStructure(response.files);
      }

      // Update preview if present
      if (response.preview) {
        setPreviewContent(response.preview);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.content,
        type: 'assistant',
        timestamp: new Date(),
        code: response.code
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat Error:', error);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        content: "An error occurred while processing your request.",
        type: 'assistant',
        timestamp: new Date()
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Navigation isExpanded={isNavExpanded} onToggle={() => setIsNavExpanded(!isNavExpanded)} />
      
      <main className={`transition-all duration-300 ${isNavExpanded ? 'ml-64' : 'ml-16'}`}>
        <div className="fixed inset-0 z-0">
          <div className="absolute inset-0 bg-black" />
          <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2 bg-blue-500/10 rounded-full filter blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-1/3 h-1/3 bg-blue-300/5 rounded-full filter blur-3xl" />
        </div>

        <div className="relative z-10 h-screen grid grid-cols-2 gap-4 p-4">
          <div className="flex flex-col space-y-4">
            <div 
              ref={chatContainerRef}
              className="flex-1 overflow-y-auto space-y-4 pr-4"
            >
              {messages.map(message => (
                <ChatMessage key={message.id} message={message} />
              ))}
            </div>
            
            <Card className="bg-black/40 backdrop-blur-sm border-gray-800 rounded-xl">
              <ChatInput
                input={input}
                setInput={setInput}
                loading={loading}
                onSubmit={handleSubmit}
                fileInputRef={fileInputRef}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                selectedMode={selectedMode}
                setSelectedMode={setSelectedMode}
              />
            </Card>
          </div>

          <div className="h-full">
            {previewContent ? (
              <ChatPreview content={previewContent} />
            ) : (
              <div className="h-full flex items-center justify-center text-gray-400">
                No preview available
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}