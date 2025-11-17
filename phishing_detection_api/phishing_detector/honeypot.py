# phishing_detector/honeypot.py

import asyncio
import sqlite3
from playwright.async_api import async_playwright
from urllib.parse import urlparse
import os

# --- Database Setup ---
# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, 'intelligence.db')

def init_honeypot_db():
    """Initializes the SQLite database for storing intelligence."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Table for domains captured by the honeypot
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS malicious_form_targets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT UNIQUE NOT NULL,
            source TEXT,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Table for domains found by the proactive hunter
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS proactive_blocklist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT UNIQUE NOT_NULL,
            source TEXT,
            xgb_score REAL,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] DB Init Error: {e}")

def save_intelligence(domain: str, source: str):
    """Saves a newly discovered malicious domain to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO malicious_form_targets (domain, source) VALUES (?, ?)",
            (domain, source)
        )
        conn.commit()
        conn.close()
        print(f"[+] Honeypot Intelligence: Saved new malicious domain: {domain}")
    except Exception as e:
        print(f"[!] Honeypot DB Error: {e}")

async def run_honeypot_analysis(url: str):
    """
    Visits a URL in a sandboxed browser, submits fake credentials,
    and captures the exfiltration domain.
    """
    print(f"--- 🍯 Honeypot Starting Analysis for: {url} ---")
    exfiltration_domain = None

    async with async_playwright() as p:
        # WARNING: This should be run in a sandboxed Docker container
        # for real-world use to prevent malware execution.
        try:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
        except Exception as e:
            print(f"❌ Honeypot Playwright Error: Failed to launch browser. {e}")
            print("    Make sure you have run: !playwright install")
            return

        try:
            # --- Network Interception ---
            def on_request(request):
                # We are looking for the POST request that sends our fake data
                if request.method == "POST" and request.post_data:
                    if "FakePassword123!" in request.post_data:
                        exfiltration_url = request.url
                        domain = urlparse(exfiltration_url).hostname
                        print(f"[!] Honeypot CAPTURED credentials being sent to: {domain}")
                        # Use a non-local var to store the domain
                        nonlocal exfiltration_domain
                        exfiltration_domain = domain

            page.on("request", on_request)

            # --- Page Interaction ---
            await page.goto(url, timeout=10000, wait_until='domcontentloaded')

            # Fill password fields
            await page.get_by_placeholder("Password").fill("FakePassword123!")
            await page.locator('input[type="password"]').fill("FakePassword123!")
            
            # Fill email/username fields
            await page.get_by_placeholder("Email").fill("fake.user@gmail.com")
            await page.get_by_placeholder("Username").fill("fake_user")
            await page.locator('input[type="email"]').fill("fake.user@gmail.com")
            
            # Click the submit button
            await page.locator('button[type="submit"], input[type="submit"]').first.click()
            
            # Wait for the network request to be captured
            await page.wait_for_timeout(3000) 

        except Exception as e:
            # Don't crash, just log the error
            print(f"[!] Honeypot Page Interaction Error: {e}")
        finally:
            await browser.close()

        # --- Save our findings ---
        if exfiltration_domain:
            save_intelligence(domain=exfiltration_domain, source=f"honeypot_from_{urlparse(url).hostname}")

    print(f"--- 🍯 Honeypot Analysis Finished for: {url} ---")

def run_honeypot_analysis_sync(url: str):
    """
    This is the synchronous wrapper function that your main.py
    will call in a background task.
    """
    try:
        asyncio.run(run_honeypot_analysis(url))
    except Exception as e:
        print(f"❌ Honeypot asyncio Error: {e}")

# --- Initialize the database when this file is imported ---
print("[*] Initializing Honeypot DB...")
init_honeypot_db()