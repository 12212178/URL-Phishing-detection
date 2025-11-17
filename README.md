# Advanced Phishing Detection API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F89939.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-13+-black.svg)](https://nextjs.org/)

This is not just a URL scanner; it's a multi-layered, active defense system for identifying and neutralizing phishing threats in real-time. It combines traditional API-based checks with a sophisticated 4-layer AI/ML ensemble, a proactive "Day Zero" threat hunter, and an active honeypot for new threat intelligence.



## 🌟 Key Features

* **Multi-Layered Analysis (L0-L4):** A deep, 5-layer analysis ensures high accuracy and resilience, from instant blocklist checks to deep AI inspection.
* **Proactive "Day Zero" Hunting (Layer 0):** A standalone service monitors Certificate Transparency (CT) logs 24/7, using heuristics to find and block phishing sites *before* they are ever visited.
* **External API Checks (Layer 1):** Fast screening against Google Safe Browsing and VirusTotal to block known threats immediately.
* **Technical Heuristic Analysis (Layer 2):** Checks for critical red flags like URL obfuscation (`@` symbol), suspicious SSL certificate age, and missing security headers.
* **Content & Brand Impersonation (Layer 3):** Scrapes and analyzes page content for common phishing tactics like forms submitting to external domains or logos hosted on suspicious URLs.
* **Advanced AI/ML Ensemble (Layer 4):** A sophisticated ensemble of three models (XGBoost, BERT, CNN-LSTM) whose predictions are fed into a final **Meta-Learner** for a highly accurate and nuanced verdict.
* **Active Defense Honeypot:** When the system finds a new, high-confidence threat, it automatically dispatches a headless browser (Playwright) to visit the site, submit fake credentials, and discover where the stolen data is being exfiltrated.

---

## 🏗️ System Architecture

The system is built on a multi-layered, "fast-fail" principle. A URL must pass through all layers to be considered safe.



1.  **Layer 0: Proactive Blocklist:** Checks against our internal, self-generated database of "Day Zero" threats found by the `proactive_hunter.py` service.
2.  **Layer 1: External APIs:** Checks the URL against Google Safe Browsing and VirusTotal.
3.  **Layer 2: Technical Heuristics:** The URL string and server headers are analyzed for obfuscation (`@`), suspicious SSL certs, and missing headers.
4.  **Layer 3: Content Analysis:** The live HTML is parsed to check for brand impersonation, suspicious forms, and external link analysis.
5.  **Layer 4: AI Ensemble:**
    * **XGBoost:** Analyzes 90+ URL features from the `dataset_B_05_2020.csv`.
    * **BERT (DistilBERT):** Performs Natural Language Processing on the extracted page text.
    * **CNN-LSTM:** A deep learning model that also analyzes the raw text for patterns.
    * **Meta-Learner:** A `LogisticRegression` model trained on the *predictions* of the three base models to make the final, weighted decision.

## Download Pre-Trained Models

Download the complete set of pre-trained models from the link below:

**[Download Pre-Trained Models](https://drive.google.com/drive/folders/156Bk39CqP0ZjpkBaGgoEa3uEAZkExV1G?usp=share_link)**

After downloading, extract the files and place them inside the following directory: phishing_detection_api/ml_models/


