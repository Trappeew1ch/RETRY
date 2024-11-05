"use client"

import { motion } from "framer-motion"
import { useState } from "react"

const userFlowSteps = [
  { id: 1, title: "Initial Query", description: "User's initial input" },
  { id: 2, title: "Requirement Analysis", description: "Gathering and refining details" },
  { id: 3, title: "Planning", description: "Defining scope, ideation, and structuring" },
  { id: 4, title: "Prototype Generation", description: "Initial code in a Markdown block" },
  { id: 5, title: "Feedback Loop", description: "User feedback, refinement, and iteration" },
  { id: 6, title: "Testing", description: "Functional checks and code optimization" },
  { id: 7, title: "Finalization", description: "Confirmation, completion, and ongoing support" },
]

export default function ProcessTimeline() {
  const [activeStep, setActiveStep] = useState(0)

  return (
    <section className="mb-16">
      <h3 className="text-2xl font-semibold mb-8 text-center">Development Process</h3>
      <div className="flex justify-center items-center space-x-4 mb-8">
        {userFlowSteps.map((step, index) => (
          <motion.div
            key={step.id}
            className={`w-12 h-12 rounded-full flex items-center justify-center cursor-pointer ${
              index <= activeStep ? "bg-white text-black" : "bg-gray-800 text-white"
            }`}
            whileHover={{ scale: 1.1 }}
            onHoverStart={() => setActiveStep(index)}
          >
            {step.id}
          </motion.div>
        ))}
      </div>
      <div className="text-center">
        <h4 className="text-xl font-semibold mb-2">{userFlowSteps[activeStep].title}</h4>
        <p className="text-gray-400">{userFlowSteps[activeStep].description}</p>
      </div>
    </section>
  )
}