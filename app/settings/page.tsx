"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Github, Google } from 'lucide-react';
import { Navigation } from '@/components/chat/Navigation';

export default function SettingsPage() {
  const [isNavExpanded, setIsNavExpanded] = useState(true);
  const [isGithubConnected, setIsGithubConnected] = useState(false);
  const [isGoogleConnected, setIsGoogleConnected] = useState(false);

  const handleGithubConnect = () => {
    // Here would be the actual GitHub OAuth flow
    setIsGithubConnected(!isGithubConnected);
  };

  const handleGoogleConnect = () => {
    // Here would be the actual Google OAuth flow
    setIsGoogleConnected(!isGoogleConnected);
  };

  return (
    <div className="min-h-screen bg-black">
      <Navigation isExpanded={isNavExpanded} onToggle={() => setIsNavExpanded(!isNavExpanded)} />
      
      <main className={`transition-all duration-300 ${isNavExpanded ? 'ml-64' : 'ml-16'}`}>
        <div className="max-w-4xl mx-auto p-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            <h1 className="text-3xl font-bold text-white">Settings</h1>

            <Card className="p-6 bg-gray-900 border-gray-800">
              <h2 className="text-xl font-semibold text-white mb-6">Connected Accounts</h2>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Github className="h-6 w-6 text-white" />
                    <div>
                      <h3 className="font-medium text-white">GitHub</h3>
                      <p className="text-sm text-gray-400">Connect to enable code sharing and repository access</p>
                    </div>
                  </div>
                  <Button
                    variant={isGithubConnected ? "destructive" : "default"}
                    onClick={handleGithubConnect}
                    className="min-w-[100px]"
                  >
                    {isGithubConnected ? 'Disconnect' : 'Connect'}
                  </Button>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Google className="h-6 w-6 text-white" />
                    <div>
                      <h3 className="font-medium text-white">Google</h3>
                      <p className="text-sm text-gray-400">Sign in with your Google account</p>
                    </div>
                  </div>
                  <Button
                    variant={isGoogleConnected ? "destructive" : "default"}
                    onClick={handleGoogleConnect}
                    className="min-w-[100px]"
                  >
                    {isGoogleConnected ? 'Disconnect' : 'Connect'}
                  </Button>
                </div>
              </div>
            </Card>

            {isGithubConnected && (
              <Card className="p-6 bg-gray-900 border-gray-800">
                <h2 className="text-xl font-semibold text-white mb-6">GitHub Integration Settings</h2>
                <div className="space-y-4">
                  <div>
                    <h3 className="font-medium text-white mb-2">Default Repository</h3>
                    <p className="text-sm text-gray-400 mb-4">Select a default repository for code exports</p>
                    <Button variant="outline" className="w-full">
                      Select Repository
                    </Button>
                  </div>
                </div>
              </Card>
            )}
          </motion.div>
        </div>
      </main>
    </div>
  );
}