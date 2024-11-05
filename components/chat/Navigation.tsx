"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { ChevronLeft, ChevronRight, Home, Plus, History, Settings } from "lucide-react";

const mainNavItems = [
  { icon: Home, label: 'Home', href: '/' },
  { icon: Plus, label: 'New Chat', href: '/chat' },
  { icon: History, label: 'Chat History', href: '/history' },
];

interface NavigationProps {
  isExpanded: boolean;
  onToggle: () => void;
}

export function Navigation({ isExpanded, onToggle }: NavigationProps) {
  return (
    <nav className={cn(
      "h-screen transition-all duration-300 border-r border-gray-800 flex flex-col bg-black/95 backdrop-blur-sm fixed left-0 top-0 z-50",
      isExpanded ? "w-64" : "w-16"
    )}>
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
          onClick={onToggle}
        >
          {isExpanded ? <ChevronLeft className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
        </Button>
      </div>

      <div className="flex-1 px-3 py-2 space-y-2">
        {mainNavItems.map((item) => {
          const Icon = item.icon;
          return (
            <Link key={item.href} href={item.href} className="block">
              <Button
                variant="ghost"
                className={cn(
                  'w-full flex items-center gap-3 justify-start hover:bg-gray-800',
                  item.href === '/chat' ? 'bg-gray-800 text-white' : 'text-gray-400',
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

      <div className="mt-auto border-t border-gray-800 pt-2 px-3 pb-3">
        <Link href="/settings">
          <Button
            variant="ghost"
            className={cn(
              'w-full flex items-center gap-3 justify-start text-gray-400 hover:bg-gray-800',
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
}