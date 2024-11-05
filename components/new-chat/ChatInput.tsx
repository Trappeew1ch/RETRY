"use client";

import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Loader2, Send, Paperclip } from 'lucide-react';
import { ModelSelect } from './ModelSelect';

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  loading: boolean;
  handleSubmit: () => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
}

export function ChatInput({
  input,
  setInput,
  loading,
  handleSubmit,
  fileInputRef
}: ChatInputProps) {
  return (
    <div className="p-4 space-y-4">
      <Textarea
        placeholder="Enter your request here..."
        className="min-h-[100px] bg-black/50 border-gray-800 resize-none"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
          }
        }}
      />

      <div className="flex flex-wrap gap-2 items-center">
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          onChange={(e) => console.log(e.target.files)}
        />
        <Button
          variant="outline"
          size="icon"
          className="w-8 h-8"
          onClick={() => fileInputRef.current?.click()}
        >
          <Paperclip className="h-4 w-4" />
        </Button>

        <ModelSelect />

        <Button
          className="ml-auto"
          onClick={handleSubmit}
          disabled={loading || !input.trim()}
        >
          {loading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </div>
    </div>
  );
}