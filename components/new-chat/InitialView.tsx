"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { ChatInput } from "./ChatInput";
import { suggestedQueries } from "./types";

interface InitialViewProps {
  input: string;
  setInput: (value: string) => void;
  loading: boolean;
  handleSubmit: () => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
}

export function InitialView({
  input,
  setInput,
  loading,
  handleSubmit,
  fileInputRef
}: InitialViewProps) {
  return (
    <motion.div
      key="initial"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, y: 20 }}
      className="max-w-2xl mx-auto p-4 relative"
    >
      <div className="text-center space-y-2 mb-8">
        <h1 className="text-3xl font-bold text-white">How can I help you?</h1>
        <p className="text-gray-400">
          Generate UI, ask questions, debug, execute code, and more.
        </p>
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

      <div className="mt-6 flex flex-wrap gap-2 justify-center">
        {suggestedQueries.map((query, index) => (
          <motion.button
            key={index}
            className="px-4 py-2 rounded-lg bg-black/40 hover:bg-black/60 border border-gray-800"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setInput(query)}
          >
            {query}
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}