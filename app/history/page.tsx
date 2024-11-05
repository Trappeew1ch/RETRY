"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Navigation } from '@/components/chat/Navigation';
import {
  MessageSquare,
  Calendar,
  Search,
  Trash2,
  ArrowUpDown,
  Download,
} from 'lucide-react';

interface ChatHistory {
  id: string;
  title: string;
  date: Date;
  preview: string;
  messages: number;
}

const mockHistory: ChatHistory[] = [
  {
    id: '1',
    title: 'React Component Development',
    date: new Date('2024-03-20'),
    preview: 'Discussion about creating reusable React components...',
    messages: 12,
  },
  {
    id: '2',
    title: 'API Integration Help',
    date: new Date('2024-03-19'),
    preview: 'Implementing REST API endpoints with authentication...',
    messages: 8,
  },
  {
    id: '3',
    title: 'Database Schema Design',
    date: new Date('2024-03-18'),
    preview: 'Planning the database structure for a new project...',
    messages: 15,
  },
];

export default function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [isNavExpanded, setIsNavExpanded] = useState(true);

  const filteredHistory = mockHistory
    .filter(
      (chat) =>
        chat.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        chat.preview.toLowerCase().includes(searchQuery.toLowerCase())
    )
    .sort((a, b) => {
      const order = sortOrder === 'asc' ? 1 : -1;
      return (b.date.getTime() - a.date.getTime()) * order;
    });

  return (
    <div className="min-h-screen bg-black">
      <Navigation isExpanded={isNavExpanded} onToggle={() => setIsNavExpanded(!isNavExpanded)} />
      
      <main className={`transition-all duration-300 ${isNavExpanded ? 'ml-64' : 'ml-16'}`}>
        <div className="max-w-4xl mx-auto p-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex items-center justify-between mb-8">
              <h1 className="text-3xl font-bold text-white">Chat History</h1>
              <div className="flex items-center gap-4">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                  className="border-gray-800"
                >
                  <ArrowUpDown className="h-4 w-4" />
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  className="border-gray-800"
                >
                  <Download className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <div className="relative mb-6">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
              <Input
                placeholder="Search chat history..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-gray-900 border-gray-800"
              />
            </div>

            <div className="space-y-4">
              {filteredHistory.map((chat) => (
                <motion.div
                  key={chat.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  whileHover={{ scale: 1.01 }}
                >
                  <Card className="p-4 bg-gray-900 border-gray-800 hover:border-gray-700 transition-colors">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h2 className="text-xl font-semibold text-white mb-2">{chat.title}</h2>
                        <p className="text-gray-400 mb-3">{chat.preview}</p>
                        <div className="flex items-center gap-4 text-sm text-gray-500">
                          <div className="flex items-center gap-1">
                            <Calendar className="h-4 w-4" />
                            {chat.date.toLocaleDateString()}
                          </div>
                          <div className="flex items-center gap-1">
                            <MessageSquare className="h-4 w-4" />
                            {chat.messages} messages
                          </div>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-gray-400 hover:text-red-400"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </Card>
                </motion.div>
              ))}

              {filteredHistory.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-gray-400">No chat history found</p>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}