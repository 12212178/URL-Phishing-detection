"use client"
import React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Shield, CheckCircle, AlertTriangle, XCircle, Scan, ServerCrash } from "lucide-react"
import { cn } from "@/lib/utils"
import { AnimatePresence, motion } from "framer-motion"

// Updated type to include the score breakdown from the API
type ScanResult = {
  verdict: "safe" | "suspicious" | "malicious"
  reasons: string[]
  url: string
  score: number
  breakdown: { [key: string]: number }
}

export default function PhishingScanner() {
  const [url, setUrl] = useState("")
  const [isScanning, setIsScanning] = useState(false)
  const [result, setResult] = useState<ScanResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleScan = async () => {
    if (!url.trim()) return
    setIsScanning(true)
    setResult(null) // Clear previous results
    setError(null)  // Clear previous errors

    try {
      // const res = await fetch("http://localhost:8000/analyze", { 
        const res = await fetch("https://kate-subsistent-distractively.ngrok-free.dev/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // --- FIX: API expects `url`, not `input` ---
        body: JSON.stringify({ url: url }),
      })

      const data = await res.json()

      if (!res.ok) {
        // Handle specific API errors, like invalid URL format
        throw new Error(data.detail || "An unexpected error occurred.")
      }

      setResult({
        ...data,
        verdict: data.verdict.toLowerCase(),
      })
    } catch (err: any) {
      console.error(err)
      setError(err.message || "Could not connect to the analysis server. Please try again later.")
    } finally {
      setIsScanning(false)
    }
  }

  const getStatusConfig = (status: ScanResult["verdict"]) => {
    switch (status) {
      case "safe":
        return {
          icon: CheckCircle,
          color: "text-green-600",
          bgColor: "bg-green-50",
          borderColor: "border-green-200",
          progressColor: "stroke-green-500",
          label: "Safe",
        }
      case "suspicious":
        return {
          icon: AlertTriangle,
          color: "text-amber-600",
          bgColor: "bg-amber-50",
          borderColor: "border-amber-200",
          progressColor: "stroke-amber-500",
          label: "Suspicious",
        }
      case "malicious":
        return {
          icon: XCircle,
          color: "text-red-600",
          bgColor: "bg-red-50",
          borderColor: "border-red-200",
          progressColor: "stroke-red-500",
          label: "Malicious",
        }
    }
  }

  // --- NEW: A component for the animated score circle ---
  const ScoreCircle = ({ score, status }: { score: number; status: ScanResult["verdict"] }) => {
    const config = getStatusConfig(status)
    const circumference = 2 * Math.PI * 45 // 45 is the radius
    const offset = circumference - (score / 100) * circumference

    return (
      <div className="relative h-28 w-28">
        <svg className="h-full w-full" viewBox="0 0 100 100">
          <circle
            className="stroke-current text-gray-200"
            strokeWidth="10"
            cx="50"
            cy="50"
            r="45"
            fill="transparent"
          />
          <motion.circle
            className={cn("origin-center -rotate-90", config.progressColor)}
            strokeWidth="10"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            cx="50"
            cy="50"
            r="45"
            fill="transparent"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={cn("text-3xl font-bold", config.color)}>{score}</span>
        </div>
      </div>
    )
  }

  // --- NEW: A component for the score breakdown ---
  const BreakdownItem = ({ label, score }: { label: string; score: number }) => (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground capitalize">{label.replace(/_/g, ' ')}</span>
      <div className="flex items-center gap-2">
        <span className="font-semibold text-foreground">{Math.round(score)}</span>
        <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
          <div className="h-full bg-primary rounded-full" style={{ width: `${Math.min(score, 100)}%` }} />
        </div>
      </div>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-2xl mx-auto space-y-8">
          <div className="text-center space-y-2">
            <Shield className="h-12 w-12 text-primary mx-auto" />
            <h1 className="text-4xl font-bold tracking-tight text-gray-900">Advanced URL Scanner</h1>
            <p className="text-lg text-muted-foreground">
              Analyze URLs for phishing threats with a multi-layered AI engine.
            </p>
          </div>

          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle>Scan a URL</CardTitle>
              <CardDescription>Enter a full URL to check for threats.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  id="url"
                  type="url"
                  placeholder="https://example.com"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleScan()}
                  disabled={isScanning}
                />
                <Button onClick={handleScan} disabled={!url.trim() || isScanning} className="min-w-fit">
                  {isScanning ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-background border-t-transparent" />
                  ) : (
                    <Scan className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          <AnimatePresence>
            {error && (
              <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <Card className="bg-red-50 border-red-200 shadow-lg">
                  <CardHeader className="flex-row items-center gap-4">
                    <ServerCrash className="h-8 w-8 text-red-600" />
                    <div>
                      <CardTitle className="text-red-800">Analysis Failed</CardTitle>
                      <CardDescription className="text-red-700">{error}</CardDescription>
                    </div>
                  </CardHeader>
                </Card>
              </motion.div>
            )}

            {result && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <Card className={cn("shadow-lg", getStatusConfig(result.verdict).bgColor, getStatusConfig(result.verdict).borderColor)}>
                  <CardHeader>
                    <div className="flex justify-between items-start">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          {React.createElement(getStatusConfig(result.verdict).icon, { className: cn("h-6 w-6", getStatusConfig(result.verdict).color) })}
                          <span className={getStatusConfig(result.verdict).color}>{getStatusConfig(result.verdict).label}</span>
                        </CardTitle>
                        <CardDescription className="pt-2 font-mono text-xs break-all">{result.url}</CardDescription>
                      </div>
                      <ScoreCircle score={result.score} status={result.verdict} />
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-foreground mb-2">Threat Indicators:</h4>
                      <ul className="space-y-1.5 text-sm text-foreground/80 list-disc pl-5">
                        {result.reasons.map((reason, index) => <li key={index}>{reason}</li>)}
                      </ul>
                    </div>
                  </CardContent>
                  {/* --- NEW: Score Breakdown Section --- */}
                  {result.breakdown && Object.keys(result.breakdown).length > 0 && (
                    <CardFooter className="bg-background/30">
                      <div className="w-full space-y-3">
                        <h4 className="font-semibold text-foreground">Score Breakdown:</h4>
                        {Object.entries(result.breakdown).map(([key, value]) => (
                          <BreakdownItem key={key} label={key} score={value} />
                        ))}
                      </div>
                    </CardFooter>
                  )}
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}