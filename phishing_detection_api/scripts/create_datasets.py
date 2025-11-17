import pandas as pd
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_page_text(html_content):
    """
    Extracts clean, visible text from raw HTML content, removing scripts,
    styles, and common non-content tags.
    """
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove all script, style, navigation, and footer elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    # Get text, replace multiple newlines/spaces, and strip leading/trailing whitespace
    text = soup.get_text(separator=' ', strip=True)
    return ' '.join(text.split())

def extract_structured_features(url, hostname, headers):
    """
    Extracts a set of numerical and boolean features from the URL, its hostname,
    and the server's response headers for the XGBoost model.
    """
    features = {}
    # URL-based features
    features['uses_https'] = 1 if url.startswith("https://") else 0
    domain_part = hostname.split('.')[-2] if '.' in hostname else hostname
    features['domain_digit_count'] = sum(c.isdigit() for c in domain_part)
    features['url_length'] = len(url)
    features['hostname_length'] = len(hostname)
    features['path_level'] = url.count('/')
    # Header-based features
    features['csp_present'] = 1 if 'Content-Security-Policy' in headers else 0
    features['hsts_present'] = 1 if 'Strict-Transport-Security' in headers else 0
    features['xfo_present'] = 1 if 'X-Frame-Options' in headers else 0
    return features

def process_url(row):
    """
    Processes a single URL to extract both text and structured features.
    This function is designed to be run in a separate thread.
    """
    url, label = row['url'], row['label']
    try:
        # Use a session for better performance and to handle cookies/headers across redirects
        with requests.Session() as session:
            response = session.get(
                url,
                timeout=8,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        hostname = urlparse(url).hostname
        if not hostname:
            return None

        # --- Data for NLP models (BERT, CNN-LSTM) ---
        text_content = extract_page_text(response.text)
        # Skip pages with minimal content as they are not useful for training
        if len(text_content) < 100:
            return None
        nlp_data = {'text': text_content, 'label': label}

        # --- Data for XGBoost model ---
        structured_data = extract_structured_features(url, hostname, response.headers)
        structured_data['is_phishing'] = label

        print(f"✅ SUCCESS: {url}")
        return {'nlp': n_data, 'structured': structured_data}

    except requests.RequestException as e:
        print(f"❌ FAILED: {url} ({type(e).__name__})")
        return None

if __name__ == "__main__":
    # Load the master list of URLs from the data directory
    try:
        url_df = pd.read_csv('../data/url_list_for_training.csv')
    except FileNotFoundError:
        print("❌ ERROR: 'data/url_list_for_training.csv' not found. Please create it first.")
        exit()

    # Shuffle the dataframe to process a mix of phishing and safe URLs
    url_df = url_df.sample(frac=1, random_state=42).reset_index(drop=True)

    nlp_records = []
    structured_records = []

    print(f"--- Starting data generation from {len(url_df)} URLs ---")

    # Use a ThreadPoolExecutor to process URLs in parallel, significantly speeding up the process
    with ThreadPoolExecutor(max_workers=12) as executor:
        # Submit all URLs to the executor
        future_to_row = {executor.submit(process_url, row): row for index, row in url_df.iterrows()}
        # Process results as they are completed
        for future in as_completed(future_to_row):
            result = future.result()
            if result:
                nlp_records.append(result['nlp'])
                structured_records.append(result['structured'])

    # Save the processed datasets to the data directory
    nlp_df = pd.DataFrame(nlp_records)
    nlp_df.to_csv('../data/phishing_text_dataset.csv', index=False)
    print(f"\n✅ NLP dataset created at 'data/phishing_text_dataset.csv' with {len(nlp_df)} rows.")

    structured_df = pd.DataFrame(structured_records)
    structured_df.to_csv('../data/structured_features_dataset.csv', index=False)
    print(f"✅ Structured features dataset created at 'data/structured_features_dataset.csv' with {len(structured_df)} rows.")