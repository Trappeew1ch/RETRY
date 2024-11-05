"use client";

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Globe,
  Smartphone,
  Terminal,
  Laptop,
  ArrowRight,
  Code,
  Braces,
  Database,
} from 'lucide-react';

const projectTypes = [
  {
    id: 'web',
    icon: Globe,
    title: 'Web Application',
    description: 'Create responsive, modern web applications',
    frameworks: ['Next.js', 'React', 'Vue', 'Angular'],
  },
  {
    id: 'mobile',
    icon: Smartphone,
    title: 'Mobile App',
    description: 'Build native mobile apps for iOS and Android',
    frameworks: ['React Native', 'Flutter', 'Native iOS', 'Native Android'],
  },
  {
    id: 'backend',
    icon: Terminal,
    title: 'Backend Service',
    description: 'Develop scalable backend services and APIs',
    frameworks: ['Node.js', 'Python', 'Go', 'Java'],
  },
  {
    id: 'desktop',
    icon: Laptop,
    title: 'Desktop Application',
    description: 'Create cross-platform desktop applications',
    frameworks: ['Electron', 'Tauri', '.NET MAUI', 'Qt'],
  },
];

const features = [
  {
    icon: Code,
    title: 'Smart Code Generation',
    description: 'AI-powered code generation with best practices',
  },
  {
    icon: Braces,
    title: 'Type Safety',
    description: 'Built-in TypeScript support for better development',
  },
  {
    icon: Database,
    title: 'Database Integration',
    description: 'Seamless integration with popular databases',
  },
];

export default function NewProject() {
  const [selectedType, setSelectedType] = useState('web');
  const [projectName, setProjectName] = useState('');
  const [step, setStep] = useState(1);

  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        stagger: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 },
  };

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        className="max-w-6xl mx-auto"
      >
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">Create New Project</h1>
          <p className="text-gray-400">
            Start your next big idea with our intelligent development platform
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            {step === 1 && (
              <motion.div variants={containerVariants}>
                <h2 className="text-2xl font-semibold mb-6">Choose Project Type</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {projectTypes.map((type) => {
                    const Icon = type.icon;
                    return (
                      <motion.div
                        key={type.id}
                        variants={itemVariants}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <Card
                          className={`p-6 cursor-pointer transition-colors ${
                            selectedType === type.id
                              ? 'bg-blue-600 border-blue-500'
                              : 'bg-gray-900 border-gray-800 hover:border-gray-700'
                          }`}
                          onClick={() => setSelectedType(type.id)}
                        >
                          <Icon className="w-8 h-8 mb-4" />
                          <h3 className="text-xl font-semibold mb-2">{type.title}</h3>
                          <p className="text-gray-400 mb-4">{type.description}</p>
                          <div className="flex flex-wrap gap-2">
                            {type.frameworks.map((framework) => (
                              <span
                                key={framework}
                                className="px-2 py-1 text-sm bg-black bg-opacity-30 rounded"
                              >
                                {framework}
                              </span>
                            ))}
                          </div>
                        </Card>
                      </motion.div>
                    );
                  })}
                </div>
              </motion.div>
            )}

            {step === 2 && (
              <motion.div variants={containerVariants}>
                <h2 className="text-2xl font-semibold mb-6">Project Details</h2>
                <Card className="p-6 bg-gray-900 border-gray-800">
                  <div className="space-y-6">
                    <div>
                      <Label htmlFor="projectName">Project Name</Label>
                      <Input
                        id="projectName"
                        value={projectName}
                        onChange={(e) => setProjectName(e.target.value)}
                        className="bg-black border-gray-800"
                        placeholder="my-awesome-project"
                      />
                    </div>

                    <Tabs defaultValue="basic" className="w-full">
                      <TabsList className="bg-black">
                        <TabsTrigger value="basic">Basic Setup</TabsTrigger>
                        <TabsTrigger value="advanced">Advanced Options</TabsTrigger>
                      </TabsList>
                      <TabsContent value="basic" className="space-y-4">
                        <div>
                          <Label>Framework Selection</Label>
                          <div className="grid grid-cols-2 gap-2 mt-2">
                            {projectTypes
                              .find((type) => type.id === selectedType)
                              ?.frameworks.map((framework) => (
                                <Button
                                  key={framework}
                                  variant="outline"
                                  className="justify-start"
                                >
                                  {framework}
                                </Button>
                              ))}
                          </div>
                        </div>
                      </TabsContent>
                      <TabsContent value="advanced" className="space-y-4">
                        <p className="text-gray-400">
                          Advanced configuration options will be available here
                        </p>
                      </TabsContent>
                    </Tabs>
                  </div>
                </Card>
              </motion.div>
            )}
          </div>

          <div className="lg:col-span-1">
            <Card className="p-6 bg-gray-900 border-gray-800 sticky top-8">
              <h3 className="text-xl font-semibold mb-4">Project Features</h3>
              <div className="space-y-4">
                {features.map((feature) => {
                  const Icon = feature.icon;
                  return (
                    <div key={feature.title} className="flex items-start space-x-3">
                      <Icon className="w-5 h-5 mt-1 text-blue-400" />
                      <div>
                        <h4 className="font-medium">{feature.title}</h4>
                        <p className="text-sm text-gray-400">{feature.description}</p>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="mt-8">
                <Button
                  className="w-full bg-blue-600 hover:bg-blue-700"
                  onClick={() => {
                    if (step === 1) setStep(2);
                    else alert('Creating project...');
                  }}
                >
                  {step === 1 ? 'Continue' : 'Create Project'}
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </div>
            </Card>
          </div>
        </div>
      </motion.div>
    </div>
  );
}