import { Message } from '@/types/chat';
import { generateChatResponse } from './google-ai';

interface ApiResponse {
  content: string;
  code?: string;
  files?: FileStructure[];
  preview?: string | null;
}

interface FileStructure {
  path: string;
  content: string;
  type: 'file' | 'directory';
}

const systemPrompt = `You are an AI assistant that helps with development tasks. When generating code:
1. If the response includes creating files or directories:
   - Use <file_structure> tags to define the file hierarchy
   - Specify full file paths and content
   - Example:
     <file_structure>
     {
       "path": "src/components/Button.tsx",
       "type": "file",
       "content": "// Button component code here"
     }
     </file_structure>

2. For HTML/CSS/JS responses:
   - Include complete, runnable code
   - Specify dependencies if needed
   - Mark code blocks with <preview> tags for auto-preview
   
3. For other responses:
   - Use clear section markers
   - Include setup instructions if needed
   - Specify environment requirements`;

export async function sendChatMessage(message: string): Promise<ApiResponse> {
  try {
    const fullPrompt = `${systemPrompt}\n\nUser: ${message}`;
    const response = await generateChatResponse(fullPrompt);
    return response;
  } catch (error) {
    console.error('Chat API Error:', error);
    throw error;
  }
}

export async function processFileStructure(files: FileStructure[]): Promise<void> {
  for (const file of files) {
    try {
      if (file.type === 'directory') {
        await fetch('/api/filesystem/create-directory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: file.path }),
        });
      } else {
        await fetch('/api/filesystem/create-file', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: file.path, content: file.content }),
        });
      }
    } catch (error) {
      console.error(`Error processing file ${file.path}:`, error);
      throw error;
    }
  }
}