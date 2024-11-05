"use client"

import { Button } from "@/components/ui/button"
import { Gift } from "lucide-react"

export default function CallToAction() {
  return (
    <section className="mt-16 text-center">
      <Button
        className="px-8 py-4 text-lg font-semibold bg-white text-black hover:bg-gray-200 transition-colors duration-300"
        onClick={() => alert("Starting your project with Retry!")}
      >
        Start Your Project
      </Button>
      <div className="mt-4 relative">
        <Gift className="w-8 h-8 mx-auto transition-all duration-300 opacity-0 hover:opacity-100 hover:text-blue-500 cursor-pointer" />
      </div>
    </section>
  )
}