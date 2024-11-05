"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Copy, MessageSquare, ThumbsUp } from "lucide-react";
import { cn } from "@/lib/utils";
import { Message } from "@/types/chat";

export function CodeBlock({ code }: { code: string }) {
  return (
    <pre className="mt-2 p-3 bg-black/30 rounded-lg overflow-x-auto">
      <code className="text-sm text-gray-300 font-mono">{code}</code>
    </pre>
  );
}

export function ChatMessage({ message }: { message: Message }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "p-4 rounded-xl backdrop-blur-sm",
        message.type === 'user' ? 'bg-blue-500/10' : 'bg-black/40'
      )}
    >
      <div className="prose prose-invert max-w-none">
        <p className="text-sm text-gray-300">{message.content}</p>
        {message.code && <CodeBlock code={message.code} />}
      </div>
      <div className="flex items-center gap-2 mt-2">
        <Button variant="ghost" size="icon" className="h-8 w-8">
          <Copy className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" className="h-8 w-8">
          <MessageSquare className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" className="h-8 w-8">
          <ThumbsUp className="h-4 w-4" />
        </Button>
      </div>
    </motion.div>
  );
}