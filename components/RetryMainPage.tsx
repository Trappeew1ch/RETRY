"use client"

import { useRef, useEffect, useState } from "react"
import Header from "./home/Header"
import ProcessTimeline from "./home/ProcessTimeline"
import PerformanceMetrics from "./home/PerformanceMetrics"
import Features from "./home/Features"
import CallToAction from "./home/CallToAction"

export default function RetryMainPage() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const backgroundRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (backgroundRef.current) {
        const { left, top, width, height } = backgroundRef.current.getBoundingClientRect()
        const x = (event.clientX - left) / width
        const y = (event.clientY - top) / height
        setMousePosition({ x, y })
      }
    }

    window.addEventListener("mousemove", handleMouseMove)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
    }
  }, [])

  return (
    <div className="min-h-screen bg-black text-white p-8 relative overflow-hidden" ref={backgroundRef}>
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `radial-gradient(circle at ${mousePosition.x * 100}% ${
            mousePosition.y * 100
          }%, rgba(59, 130, 246, 0.15), transparent 30%)`,
        }}
      />
      <Header />
      <ProcessTimeline />
      <PerformanceMetrics />
      <Features />
      <CallToAction />
    </div>
  )
}