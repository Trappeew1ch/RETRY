import { NextResponse } from 'next/server';
import { mkdir } from 'fs/promises';
import { join } from 'path';

export async function POST(request: Request) {
  try {
    const { path } = await request.json();
    const fullPath = join(process.cwd(), path);
    
    await mkdir(fullPath, { recursive: true });
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Create Directory Error:', error);
    return NextResponse.json(
      { error: 'Failed to create directory' },
      { status: 500 }
    );
  }
}