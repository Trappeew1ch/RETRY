"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Code, Zap, RefreshCw, Scale } from "lucide-react"

const features = [
  { 
    icon: Code, 
    title: "Interactive Development", 
    description: "Real-time code previews, scenario-based execution, and functional testing",
    details: "Experience coding like never before with instant feedback and contextual suggestions. Our platform adapts to your coding style and project requirements in real-time."
  },
  { 
    icon: Zap, 
    title: "Advanced LLM Integration", 
    description: "Utilizing state-of-the-art language models",
    details: "Harness the power of cutting-edge AI to assist in code generation, bug detection, and optimization. Our LLMs understand context and intent, providing intelligent suggestions throughout your development process."
  },
  { 
    icon: RefreshCw, 
    title: "Iterative Improvement", 
    description: "Unique iterative process and feedback integration",
    details: "Continuously refine your code with our intelligent feedback loop. Retry learns from each iteration, helping you write better, more efficient code with every cycle."
  },
  { 
    icon: Scale, 
    title: "Scalable Support", 
    description: "Continued assistance and customization even after project completion",
    details: "Our support doesn't end when your project is live. Retry grows with your application, offering ongoing optimizations, security updates, and feature suggestions tailored to your evolving needs."
  },
]

export default function Features() {
  return (
    <section>
      <h3 className="text-2xl font-semibold mb-8 text-center">Key Features</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {features.map((feature, index) => (
          <Card key={index} className="bg-black border border-gray-800">
            <CardHeader>
              <feature.icon className="w-8 h-8 mb-2 text-white" />
              <CardTitle className="text-white">{feature.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-400 mb-4">{feature.description}</p>
              <p className="text-white">{feature.details}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  )
}