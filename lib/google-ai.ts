import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI('AIzaSyDBq1kqC-C_i69Ur-RR48eDKi4XsIqOKCs');

export async function generateChatResponse(prompt: string) {
  try {
    const model = genAI.getGenerativeModel({ model: 'gemini-pro' });
    
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    return {
      content: text,
      files: extractFileStructure(text),
      preview: extractPreviewContent(text),
    };
  } catch (error) {
    console.error('Google AI Error:', error);
    throw error;
  }
}

function extractFileStructure(response: string): any[] {
  const fileStructureMatch = response.match(/<file_structure>([\s\S]*?)<\/file_structure>/);
  if (fileStructureMatch) {
    try {
      return JSON.parse(fileStructureMatch[1]);
    } catch (error) {
      console.error('Failed to parse file structure:', error);
      return [];
    }
  }
  return [];
}

function extractPreviewContent(response: string): string | null {
  const previewMatch = response.match(/<preview>([\s\S]*?)<\/preview>/);
  return previewMatch ? previewMatch[1] : null;
}