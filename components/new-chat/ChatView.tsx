"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Maximize2 } from "lucide-react";
import { Message as MessageComponent } from "./Message";
import { ChatInput } from "./ChatInput";
import { Message } from "./types";
import { CodeBlock } from "./CodeBlock";

interface ChatViewProps {
  messages: Message[];
  input: string;
  setInput: (value: string) => void;
  loading: boolean;
  handleSubmit: () => void;
  chatContainerRef: React.RefObject<HTMLDivElement>;
  fileInputRef: React.RefObject<HTMLInputElement>;
}

export function ChatView({
  messages,
  input,
  setInput,
  loading,
  handleSubmit,
  chatContainerRef,
  fileInputRef
}: ChatViewProps) {
  return (
    <motion.div
      key="split"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-full flex"
    >
      <div className="flex-1 p-4 border-r border-gray-800 overflow-hidden flex flex-col">
        <div 
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto space-y-4 mb-4 pr-4"
        >
          {messages.map(message => (
            <MessageComponent key={message.id} message={message} />
          ))}
        </div>
        
        <Card className="bg-black/40 backdrop-blur-sm border-gray-800 rounded-xl">
          <ChatInput
            input={input}
            setInput={setInput}
            loading={loading}
            handleSubmit={handleSubmit}
            fileInputRef={fileInputRef}
          />
        </Card>
      </div>

      <div className="w-1/2 p-4 relative">
        <div className="absolute top-4 right-4">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-gray-400 hover:text-white"
          >
            <Maximize2 className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="mt-8">
          {messages[messages.length - 1]?.code && (
            <CodeBlock code={messages[messages.length - 1].code} />
          )}
        </div>
      </div>
    </motion.div>
  );
}