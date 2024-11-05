"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useState } from "react";
import { models, modes } from "./types";

export function ModelSelect() {
  const [selectedModel, setSelectedModel] = useState<typeof models[number]>("gemini-1.5-pro-002");
  const [selectedMode, setSelectedMode] = useState<typeof modes[number]['value']>("basic");

  return (
    <>
      <Select value={selectedModel} onValueChange={(value: typeof models[number]) => setSelectedModel(value)}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Select Model" />
        </SelectTrigger>
        <SelectContent>
          {models.map((model) => (
            <SelectItem key={model} value={model}>
              {model}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select value={selectedMode} onValueChange={(value: typeof modes[number]['value']) => setSelectedMode(value)}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Select Mode" />
        </SelectTrigger>
        <SelectContent>
          {modes.map((mode) => (
            <SelectItem key={mode.value} value={mode.value}>
              {mode.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </>
  );
}