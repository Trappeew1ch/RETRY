"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Copy, MessageSquare, ThumbsUp, Loader2 } from "lucide-react";
import { CodeBlock } from "./CodeBlock";
import { Message as MessageType } from "./types";

export function Message({ message }: { message: MessageType }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-4 rounded-xl ${
        message.type === 'user' ? 'bg-blue-500/10' : 'bg-black/40'
      } backdrop-blur-sm`}
    >
      <div className="prose prose-invert max-w-none">
        {message.status ? (
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm text-gray-300">
              {message.status === 'thinking' ? 'Analyzing request...' :
               message.status === 'generating' ? 'Generating response...' :
               'Completing...'}
            </span>
          </div>
        ) : (
          <>
            <p className="text-sm text-gray-300">{message.content}</p>
            {message.code && <CodeBlock code={message.code} />}
          </>
        )}
      </div>
      {!message.status && (
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
      )}
    </motion.div>
  );
}