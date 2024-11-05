export interface Message {
  id: string;
  content: string;
  type: 'user' | 'assistant';
  timestamp: Date;
  code?: string;
}

export const models = [
  'gemini-1.5-pro-002',
  'gemini-1.5-flash-002',
  'gemini-1.5-flash-001',
  'gemini-1.5-pro-exp-0827',
  'gemini-1.5-flash-exp-0827',
  'gemini-1.5-flash-8b-exp-0827',
  'Qwen/Qwen2.5-72B-Instruct'
] as const;

export const modes = [
  { value: 'basic', label: 'Basic Mode' },
  { value: 'pro', label: 'Pro Mode' },
  { value: 'model-to-model', label: 'Model to Model Mode' }
] as const;

export const suggestedQueries = [
  "Сгенерировать пошаговый алгоритм →",
  "Как структурировать вывод LLM →",
  "Функция для выравнивания вложенных массивов →"
];