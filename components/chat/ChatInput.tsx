"use client";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, Send, Paperclip, Github } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { models, modes } from "@/types/chat";

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  loading: boolean;
  onSubmit: () => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  selectedModel: typeof models[number];
  setSelectedModel: (value: typeof models[number]) => void;
  selectedMode: typeof modes[number]['value'];
  setSelectedMode: (value: typeof modes[number]['value']) => void;
}

export function ChatInput({
  input,
  setInput,
  loading,
  onSubmit,
  fileInputRef,
  selectedModel,
  setSelectedModel,
  selectedMode,
  setSelectedMode,
}: ChatInputProps) {
  return (
    <div className="flex flex-col space-y-3">
      <Textarea
        placeholder="Введите ваш запрос здесь..."
        className="min-h-[60px] bg-black/50 border-gray-800 resize-none text-white placeholder-gray-400 rounded-xl text-sm"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
        }}
      />
      
      <div className="flex flex-wrap gap-2 items-center">
        <Button
          size="icon"
          className="w-8 h-8 bg-black/50 hover:bg-black/70 text-white rounded-lg border border-gray-800"
          onClick={onSubmit}
          disabled={loading || !input.trim()}
        >
          {loading ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <Send className="w-3 h-3" />
          )}
        </Button>

        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          onChange={(e) => console.log(e.target.files)}
        />
        <Button
          variant="outline"
          size="icon"
          className="w-8 h-8 rounded-lg bg-black/50 border-gray-800 text-white hover:text-white hover:bg-black/70"
          onClick={() => fileInputRef.current?.click()}
        >
          <Paperclip className="w-3 h-3" />
        </Button>

        <Button
          variant="outline"
          size="icon"
          className="w-8 h-8 rounded-lg bg-black/50 border-gray-800 text-white hover:text-white hover:bg-black/70"
        >
          <Github className="w-3 h-3" />
        </Button>

        <Select value={selectedModel} onValueChange={setSelectedModel}>
          <SelectTrigger className="w-[160px] h-8 bg-black/50 border-gray-800 text-white rounded-lg text-xs hover:bg-black/70">
            <SelectValue placeholder="Выберите модель" />
          </SelectTrigger>
          <SelectContent className="bg-black/90 border-gray-800 text-white">
            {models.map((model) => (
              <SelectItem key={model} value={model} className="text-white text-xs">
                {model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={selectedMode} onValueChange={setSelectedMode}>
          <SelectTrigger className="w-[160px] h-8 bg-black/50 border-gray-800 text-white rounded-lg text-xs hover:bg-black/70">
            <SelectValue placeholder="Выберите режим" />
          </SelectTrigger>
          <SelectContent className="bg-black/90 border-gray-800 text-white">
            {modes.map((mode) => (
              <SelectItem key={mode.value} value={mode.value} className="text-white text-xs">
                {mode.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}