"use client";

import { useEffect, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Maximize2 } from 'lucide-react';

interface ChatPreviewProps {
  content: string;
}

export function ChatPreview({ content }: ChatPreviewProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    if (iframeRef.current) {
      const iframe = iframeRef.current;
      const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
      
      if (iframeDoc) {
        iframeDoc.open();
        iframeDoc.write(content);
        iframeDoc.close();
      }
    }
  }, [content]);

  return (
    <Card className="relative h-full bg-gray-900 border-gray-800">
      <div className="absolute top-4 right-4 z-10">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-gray-400 hover:text-white"
        >
          <Maximize2 className="h-4 w-4" />
        </Button>
      </div>
      <iframe
        ref={iframeRef}
        className="w-full h-full bg-white rounded-lg"
        sandbox="allow-scripts"
        title="Preview"
      />
    </Card>
  );
}