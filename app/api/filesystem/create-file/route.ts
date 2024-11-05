import { NextResponse } from 'next/server';
import { writeFile } from 'fs/promises';
import { join, dirname } from 'path';
import { mkdir } from 'fs/promises';

export async function POST(request: Request) {
  try {
    const { path, content } = await request.json();
    const fullPath = join(process.cwd(), path);
    
    // Ensure directory exists
    await mkdir(dirname(fullPath), { recursive: true });
    
    // Write file
    await writeFile(fullPath, content);
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Create File Error:', error);
    return NextResponse.json(
      { error: 'Failed to create file' },
      { status: 500 }
    );
  }
}