import { NextResponse } from 'next/server';
import { generateChatResponse } from '@/lib/google-ai';

export async function POST(request: Request) {
  try {
    const { message, systemPrompt } = await request.json();
    const fullPrompt = `${systemPrompt}\n\nUser: ${message}`;
    
    const response = await generateChatResponse(fullPrompt);

    return NextResponse.json(response);
  } catch (error) {
    console.error('Chat API Error:', error);
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    );
  }
}