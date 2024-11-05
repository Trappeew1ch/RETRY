"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

const llmComparisonData = [
  { name: "Claude 3.5.1 Sonnet", previousScore: 50, improvement: 37, currentScore: 87 },
  { name: "Qwen 2.5 72B", previousScore: 40, improvement: 34, currentScore: 74 },
  { name: "ChatGPT O1", previousScore: 45, improvement: 34, currentScore: 79 },
]

export default function PerformanceMetrics() {
  return (
    <section className="mb-16">
      <h3 className="text-2xl font-semibold mb-8 text-center">Performance Comparison</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card className="bg-black border border-gray-800">
          <CardHeader>
            <CardTitle className="text-white">SWE Bench Performance</CardTitle>
            <CardDescription className="text-gray-400">Retry vs Other Agents</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col lg:flex-row items-center">
            <div className="relative w-64 h-64 mb-4 lg:mb-0">
              <svg className="w-full h-full" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#333" strokeWidth="10" />
                <circle
                  cx="50"
                  cy="50"
                  r="45"
                  fill="none"
                  stroke="#fff"
                  strokeWidth="10"
                  strokeDasharray="282.7"
                  strokeDashoffset="90.46"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-4xl font-bold text-white">68</span>
              </div>
            </div>
            <div className="lg:ml-8 text-center lg:text-left">
              <h4 className="text-xl font-semibold mb-2">Understanding SWE Bench</h4>
              <p className="text-gray-400">
                SWE Bench is a standardized metric for evaluating AI coding assistants. Retry's score of 68 is twice the industry average, showcasing its superior performance in code generation, problem-solving, and software engineering tasks.
              </p>
              <p className="text-white mt-4">
                Retry's score is double that of other agents, demonstrating its unparalleled capabilities in assisting developers across various programming challenges.
              </p>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-black border border-gray-800">
          <CardHeader>
            <CardTitle className="text-white">LLM Improvement Rates</CardTitle>
            <CardDescription className="text-gray-400">Coding Task Performance Increase</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col lg:flex-row items-center">
            <div className="lg:w-1/3 mb-4 lg:mb-0">
              <h4 className="text-xl font-semibold mb-2">Rapid Progress</h4>
              <p className="text-gray-400">
                This chart illustrates the significant improvements in coding task performance for leading language models. The bars represent the increase in performance scores, showcasing Retry's commitment to leveraging cutting-edge AI advancements.
              </p>
            </div>
            <div className="lg:w-2/3 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={llmComparisonData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis type="number" stroke="#fff" />
                  <YAxis dataKey="name" type="category" stroke="#fff" />
                  <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }} labelStyle={{ color: '#fff' }} />
                  <Legend wrapperStyle={{ color: '#fff' }} />
                  <Bar dataKey="previousScore" stackId="a" fill="#333" name="Previous Score" />
                  <Bar dataKey="improvement" stackId="a" fill="#fff" name="Improvement" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}