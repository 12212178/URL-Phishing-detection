# phishing_detector/analysis_layers.py (Corrected and Final Version)

import os
import joblib
import requests
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import base64
from collections import Counter
import tldextract  # <-- NEW IMPORT for robust domain parsing

import ssl
import socket
from datetime import datetime

# --- Global variables ---
XGB_MODEL, META_LEARNER, NLP_CLASSIFIER, CNN_LSTM_MODEL, CNN_TOKENIZER = (None,) * 5
XGB_FEATURES = None

def load_all_models():
    """Loads all trained models from the ml_models directory into memory."""
    global XGB_MODEL, META_LEARNER, NLP_CLASSIFIER, CNN_LSTM_MODEL, CNN_TOKENIZER, XGB_FEATURES
    print("[*] Attempting to load all Layer 4 models into memory...")
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, '..', 'ml_models')
        
        if not os.path.exists(model_path):
            print(f"❌ CRITICAL ERROR: Model directory not found. Are you running from the project root?")
            return

        XGB_MODEL = joblib.load(os.path.join(model_path, 'xgboost_model.joblib'))
        META_LEARNER = joblib.load(os.path.join(model_path, 'meta_learner.joblib'))
        CNN_LSTM_MODEL = load_model(os.path.join(model_path, 'cnn_lstm_model.h5'))
        CNN_TOKENIZER = joblib.load(os.path.join(model_path, 'cnn_lstm_tokenizer.joblib'))
        bert_path = os.path.join(model_path, 'my_finetuned_phishing_model')
        NLP_CLASSIFIER = pipeline("text-classification", model=bert_path, device=-1)
        
        XGB_FEATURES = XGB_MODEL.get_booster().feature_names
        print(f"    -> XGBoost model expects {len(XGB_FEATURES)} features.")
        print("✅ All models loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load one or more models. Error: {e}")

# --- Helper Functions ---
def extract_page_text(html_content):
    if not html_content: return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]): element.decompose()
    return ' '.join(soup.get_text(separator=' ', strip=True).split())

def extract_live_subset_features(url, hostname):
    features = {'url_len': len(url), 'domain_len': len(hostname), 'nb_dots': url.count('.'), 'nb_slash': url.count('/'), 'nb_hyphens': url.count('-'), 'httpss': 1 if url.startswith("https://") else 0}
    return features

def get_ssl_certificate_info(hostname):
    """
    Connects to the host and retrieves its SSL certificate details.
    Returns the certificate dictionary or None on failure.
    """
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=4) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as sslsock:
                cert = sslsock.getpeercert()
                return cert
    except (socket.timeout, ssl.SSLError, ConnectionRefusedError, socket.gaierror, OSError):
        # Handle connection failures or sites without SSL
        return None
    except Exception as e:
        # Catch other potential errors
        print(f"[!] SSL Check Error for {hostname}: {e}")
        return None

# --- Analysis Layers ---
def analyze_layer1_apis(analysis_context):
    """Layer 1: Fast Screening using external APIs (Google Safe Browsing & VirusTotal)."""
    url = analysis_context['url']
    total_score, all_reasons = 0, []

    # 1a. Google Safe Browsing Check
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: all_reasons.append("Google Safe Browsing API key not configured.")
        else:
            gsb_url = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
            payload = { "client": {"clientId": "phishing-scanner", "clientVersion": "1.0"}, "threatInfo": { "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"], "platformTypes": ["ANY_PLATFORM"], "threatEntryTypes": ["URL"], "threatEntries": [{"url": url}] } }
            gsb_res = requests.post(gsb_url, json=payload, timeout=4).json()
            if "matches" in gsb_res:
                total_score = 95; all_reasons.append("Flagged by Google Safe Browsing as a known malicious site.")
    except requests.RequestException: all_reasons.append("Google Safe Browsing check could not be completed.")

    # 1b. VirusTotal Check
    try:
        api_key = os.getenv("VT_API_KEY")
        if not api_key: all_reasons.append("VirusTotal API key not configured.")
        else:
            url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
            vt_url = f"https://www.virustotal.com/api/v3/urls/{url_id}"
            headers = {"x-apikey": api_key}
            vt_res = requests.get(vt_url, headers=headers, timeout=4)
            if vt_res.status_code == 200:
                data = vt_res.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                malicious_count = data.get("malicious", 0)
                if malicious_count > 0:
                    vt_score = min(80, 10 + (malicious_count * 8))
                    total_score = max(total_score, vt_score)
                    all_reasons.append(f"Flagged by {malicious_count} engines on VirusTotal.")
    except requests.RequestException: all_reasons.append("VirusTotal check could not be completed.")
    return total_score, all_reasons

def analyze_layer2_technical(analysis_context):
    """Layer 2: Technical Heuristics (Security Headers)."""
    score, reasons = 0, []
    url = analysis_context['url']
    hostname = analysis_context['hostname']
    headers = analysis_context['response'].headers
    
    netloc = urlparse(url).netloc

    # 1. Check for the critical '@' obfuscation attack
    if '@' in netloc:
        score += 80  # High, immediate penalty
        reasons.append("URL contains an '@' symbol in its network location, a critical indicator of a URL obfuscation attack.")
    
    # 2. Check for all OTHER suspicious characters
    # We remove '@' from this list because it's handled above
    SUSPICIOUS_CHARS = {'_', '~', '!', '$', '&', '*', '(', ')', '=', '+', ',', ';', ':', '?', '/'}
    
    # Check for these characters in the hostname only
    found_chars = {char for char in hostname if char in SUSPICIOUS_CHARS}
    
    if found_chars:
        score += 30 # A lower, but still serious penalty
        char_list = ", ".join(f"'{c}'" for c in sorted(list(found_chars)))
        reasons.append(f"Domain name contains suspicious character(s): {char_list}, a common URL obfuscation tactic.")
    # --- [END OF FIX] ---

    
    # --- [IMPROVED LOGIC] Suspicious Digits in Domain Check ---
    try:
        domain_part = tldextract.extract(hostname).domain
        if domain_part:
            digit_count = sum(c.isdigit() for c in domain_part)
            if digit_count >= 2:
                score += 15
                reasons.append(f"Main domain name ('{domain_part}') contains {digit_count} digits, a common phishing tactic.")
    except Exception:
        pass
    
    # --- Header and Protocol Checks ---
    if 'Content-Security-Policy' not in headers:
        score += 10; reasons.append("Missing Content-Security-Policy (CSP) header.")
    if 'X-Frame-Options' not in headers:
        score += 10; reasons.append("Missing X-Frame-Options header (Clickjacking risk).")
    if 'Strict-Transport-Security' not in headers and url.startswith("https://"):
        score += 10; reasons.append("Missing HSTS header (Downgrade attack risk).")
    if not url.startswith("https://"):
        score += 25; reasons.append("URL uses insecure HTTP protocol.")
    
    if url.startswith("https://"):
        cert_info = get_ssl_certificate_info(hostname)
        if cert_info:
            try:
                # 1. Check Certificate Age
                # Format: 'Nov 10 10:05:15 2025 GMT'
                date_format = '%b %d %H:%M:%S %Y %Z'
                start_date = datetime.strptime(cert_info['notBefore'], date_format)
                end_date = datetime.strptime(cert_info['notAfter'], date_format)
                
                days_active = (datetime.now() - start_date).days
                
                if days_active <= 7: # Created in the last week
                    score += 20
                    reasons.append(f"SSL certificate is brand new ({days_active} days old), a common tactic for disposable phishing sites.")

                # 2. Check Certificate Duration (90-day certs are common for phishing)
                total_duration_days = (end_date - start_date).days

                if total_duration_days <= 91: # 90 days is standard for Let's Encrypt
                    score += 10
                    reasons.append(f"SSL certificate has a short validity period ({total_duration_days} days), common for automated, free, and disposable sites.")
            
            except Exception as e:
                print(f"[!] SSL Date Parse Error: {e}")
                
        else:
            # This is suspicious: an HTTPS site that we can't get cert info from
            score += 15
            reasons.append("Failed to retrieve SSL certificate details for an HTTPS site.")
    
    else:
        # This is the original check, it will run if the URL is not HTTPS
        score += 25; reasons.append("URL uses insecure HTTP protocol.")

    return score, reasons

def analyze_layer3_content(analysis_context):
    """Layer 3: Brand Impersonation Analysis."""
    score, reasons = 0, []
    hostname = analysis_context['hostname']
    response_text = analysis_context['response'].text
    soup = BeautifulSoup(response_text, 'html.parser')

    # 1. Most common link domain analysis
    links = [a.get('href') for a in soup.find_all('a', href=True)]
    domains = [urlparse(link).hostname for link in links if urlparse(link).hostname and '.' in urlparse(link).hostname]
    if domains:
        most_common_domain = Counter(domains).most_common(1)[0][0]
        # Check if the most common domain is a subdomain of the current host (e.g., help.example.com vs example.com)
        if not most_common_domain.endswith(hostname):
            score += 25
            reasons.append(f"Most links on the page point to an external domain ({most_common_domain}).")
    
    # 2. Logo Domain analysis
    for img in soup.find_all('img', src=True):
        if 'logo' in img['src'].lower():
            logo_host = urlparse(img['src']).hostname
            if logo_host and not logo_host.endswith(hostname):
                score += 20
                reasons.append("The site logo is hosted on an external domain.")
                break # Only flag once
    
    # 3. Form action Domain validation
    for form in soup.find_all('form', action=True):
        action_host = urlparse(form['action']).hostname
        if action_host and not action_host.endswith(hostname):
            score += 50 # High penalty for this
            reasons.append("A form on the page submits data to a different domain, a major phishing indicator.")
            break # Only flag once

    return score, reasons

def analyze_layer4_ai(analysis_context):
    """Layer 4: Advanced AI/ML Ensemble Analysis."""
    if not all([XGB_MODEL, XGB_FEATURES, META_LEARNER, NLP_CLASSIFIER, CNN_LSTM_MODEL, CNN_TOKENIZER]):
        return 0, ["Layer 4 models not loaded; advanced analysis skipped."]
    try:
        url, hostname, response = analysis_context['url'], analysis_context['hostname'], analysis_context['response']

        # 1. Prepare XGBoost features
        live_features_template = pd.DataFrame(0, index=[0], columns=XGB_FEATURES)
        live_subset = extract_live_subset_features(url, hostname)
        for feature, value in live_subset.items():
            if feature in live_features_template.columns: live_features_template[feature] = value
        
        xgb_pred_proba = XGB_MODEL.predict_proba(live_features_template)[:, 1][0]

        # 2. Prepare NLP features
        page_text = extract_page_text(response.text)
        
        # BERT prediction
        bert_result = NLP_CLASSIFIER(page_text, truncation=True, max_length=512)[0]
        bert_pred_proba = bert_result['score'] if bert_result['label'] == 'LABEL_1' else 1 - bert_result['score']
        
        # CNN-LSTM prediction
        sequences = CNN_TOKENIZER.texts_to_sequences([page_text])
        padded = pad_sequences(sequences, maxlen=200)
        cnn_pred_proba = CNN_LSTM_MODEL.predict(padded, verbose=0).flatten()[0]
        
        # 3. Get Meta-Learner prediction
        # This structure is correct for creating a single-row DataFrame
        meta_data = {
            'xgb_pred': float(xgb_pred_proba), 
            'bert_pred': float(bert_pred_proba), 
            'cnn_pred': float(cnn_pred_proba)
        }
        meta_input = pd.DataFrame([meta_data])
        final_proba = META_LEARNER.predict_proba(meta_input)[:, 1][0]
        
        print("\n--- AI Layer 4 Debug Info ---")
        print(f"XGBoost Phishing Probability: {xgb_pred_proba:.2f}")
        print(f"BERT Phishing Probability:    {bert_pred_proba:.2f}")
        print(f"CNN-LSTM Phishing Probability:{cnn_pred_proba:.2f}")
        print(f"Meta-Learner Final Probability: {final_proba:.2f}")
        
        score, reasons = 0, []
        if final_proba > 0.75:
            score = 50 + (final_proba * 50)
            reasons.append(f"Advanced AI ensemble predicts phishing with {final_proba:.0%} confidence.")
        return score, reasons
    
    except Exception as e:
        print(f"---! AI LAYER 4 CRASHED !---")
        print(f"The specific error is: {e}")
        return 0, [f"Layer 4 analysis encountered an error: {e}"]

# --- Orchestrator Function ---
def run_full_analysis(url, response):
    """Runs all analysis layers and calculates a final risk score."""
    analysis_context = {'url': url, 'hostname': urlparse(url).hostname, 'response': response, 'soup': BeautifulSoup(response.text, 'html.parser')}
    all_reasons, score_breakdown = [], {}

    # Layer 1: Fast exit for known malicious sites
    l1_score, l1_reasons = analyze_layer1_apis(analysis_context)
    score_breakdown['layer_1_api_checks'] = l1_score
    all_reasons.extend(l1_reasons)
    if l1_score > 90: 
        return 100, all_reasons, score_breakdown

    # Layers 2 & 3: Heuristic checks
    l2_score, l2_reasons = analyze_layer2_technical(analysis_context)
    l3_score, l3_reasons = analyze_layer3_content(analysis_context)
    score_breakdown['layer_2_technical_heuristics'] = l2_score
    score_breakdown['layer_3_content_heuristics'] = l3_score
    all_reasons.extend(l2_reasons)
    all_reasons.extend(l3_reasons)

    # Layer 4: AI/ML deep analysis
    l4_score, l4_reasons = analyze_layer4_ai(analysis_context)
    score_breakdown['layer_4_ai_ensemble'] = round(l4_score, 2)
    all_reasons.extend(l4_reasons)
    
    # Calculate final cumulative score
    total_score = l1_score + l2_score + l3_score + l4_score
    final_score = min(100, int(total_score))

    return final_score, all_reasons, score_breakdown