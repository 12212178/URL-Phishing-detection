import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from urllib.parse import urlparse
import httpx

#
# ❗️ ALL THE SECRET-LOADING CODE (try...except) HAS BEEN REMOVED. ❗️
# The environment variables are set by the runner script.
#

# This middleware is needed to allow web browsers to access your API
from fastapi.middleware.cors import CORSMiddleware

# Use a relative import to bring in the analysis logic from the same package
from .analysis_layers import load_all_models, run_full_analysis
# Import the synchronous wrapper for our honeypot
from .honeypot import run_honeypot_analysis_sync

# Initialize the FastAPI application
app = FastAPI(
    title="Advanced Phishing Detection API",
    description="A multi-layered API that uses heuristics and a deep learning ensemble to detect phishing URLs.",
    version="2.0"
)

# --- [UPDATED] Add CORS Middleware ---
origins = [
    "http://localhost:3000", # Example for a React frontend
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.ngrok-free\.dev", # ALLOWS ALL NGROK URLS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- [END UPDATED] ---

# --- Application Startup Event ---
@app.on_event("startup")
def startup_event():
    """Load all ML models into memory once when the application starts."""
    load_all_models()

# --- Pydantic Model for API Request Body ---
class AnalyzeRequest(BaseModel):
    url: str

# --- Main API Endpoint ---
@app.post("/analyze", summary="Analyze a URL for phishing threats")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks): 
    """
    Analyzes a given URL through a multi-layered detection engine.
    """
    url = request.url.strip()
    try:
        # Basic validation before making any network requests
        parsed_url = urlparse(url)
        if not (parsed_url.scheme and parsed_url.netloc):
            raise ValueError("Invalid URL structure. Must include a scheme (e.g., https://).")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid URL format provided.")

    # --- [UPGRADE] Perform the network request asynchronously ---
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Use an async client to make the request without blocking the server
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(
                url,
                timeout=7,
                headers=headers
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
    except (httpx.RequestError, httpx.TimeoutException) as e:
        # If we can't fetch the URL, it's a major red flag.
        return {
            "url": url,
            "verdict": "Suspicious",
            "score": 60,
            "reasons": [f"Failed to fetch webpage content, a major red flag. Reason: {type(e).__name__}"],
            "breakdown": {"fetch_error": 60}
        }
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    # --- Hand off to the analysis orchestrator in analysis_layers.py ---
    final_score, all_reasons, score_breakdown = run_full_analysis(url, response)

    # --- Determine Final Verdict based on the calculated score ---
    if final_score >= 70:
        verdict = "Malicious"
    elif final_score >= 35:
        verdict = "Suspicious"
    else:
        verdict = "Safe"

    # --- [NEW] ACTIVE DEFENSE & HONEYPOT TRIGGER ---
    # Check if the AI/ML layers found a new threat (not found by Layer 1 APIs)
    is_high_confidence = (final_score >= 80)
    is_new_discovery = (score_breakdown.get('layer_1_api_checks', 0) < 10) # Checks if L1 score is low

    if is_high_confidence and is_new_discovery:
        print(f"[*] Honeypot Triggered: Sending {url} for active analysis.")
        # Add the honeypot task to run in the background
        # We use the 'sync' wrapper function we created in honeypot.py
        background_tasks.add_task(run_honeypot_analysis_sync, url)
    # --- END OF NEW SECTION ---

    # --- Return the complete analysis result ---
    return {
        "url": url,
        "verdict": verdict,
        "score": final_score,
        "reasons": all_reasons if all_reasons else ["No suspicious indicators found."],
        "breakdown": score_breakdown,
    }