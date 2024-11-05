"use client";

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  MessageSquare,
  History,
  Settings,
  ChevronRight,
  ChevronLeft,
  Home
} from 'lucide-react';

const Navigation = () => {
  const pathname = usePathname();
  const [isExpanded, setIsExpanded] = useState(true);

  const navItems = [
    {
      icon: Home,
      label: 'Home',
      href: '/',
    },
    {
      icon: MessageSquare,
      label: 'New Chat',
      href: '/chat',
    },
    {
      icon: History,
      label: 'History',
      href: '/history',
    },
  ];

  return (
    <nav 
      className={cn(
        "h-screen transition-all duration-300 border-r border-gray-800 flex flex-col bg-black",
        isExpanded ? "w-64" : "w-16"
      )}
    >
      <div className="flex items-center justify-between p-4">
        {isExpanded && (
          <Link href="/" className="transition-opacity">
            <h1 className="text-2xl font-bold text-white">Retry</h1>
          </Link>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="text-gray-400 hover:text-white"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? <ChevronLeft className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
        </Button>
      </div>

      <div className="flex-1 px-3 py-2 space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;
          return (
            <Link key={item.href} href={item.href} className="block">
              <Button
                variant="ghost"
                className={cn(
                  'w-full flex items-center gap-3 justify-start hover:bg-gray-800',
                  isActive ? 'bg-gray-800 text-white' : 'text-gray-400',
                  !isExpanded && 'justify-center'
                )}
              >
                <Icon className="h-5 w-5 flex-shrink-0" />
                {isExpanded && <span>{item.label}</span>}
              </Button>
            </Link>
          );
        })}
      </div>

      <div className="p-3 mt-auto">
        <Link href="/settings" className="block">
          <Button
            variant="ghost"
            className={cn(
              'w-full flex items-center gap-3 justify-start text-gray-400 hover:bg-gray-800',
              pathname === '/settings' && 'bg-gray-800 text-white',
              !isExpanded && 'justify-center'
            )}
          >
            <Settings className="h-5 w-5 flex-shrink-0" />
            {isExpanded && <span>Settings</span>}
          </Button>
        </Link>
      </div>
    </nav>
  );
};

export default Navigation;