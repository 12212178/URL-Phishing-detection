# format_kaggle_data.py
import pandas as pd
from urllib.parse import urlparse

# --- Configuration ---
# The name of the file you downloaded from Kaggle
KAGGLE_INPUT_FILE = 'data/phishing_website_dataset.csv'

# The names of the two files we will create
NLP_OUTPUT_FILE = 'data/phishing_text_dataset.csv'
STRUCTURED_OUTPUT_FILE = 'data/structured_features_dataset.csv'

def format_data():
    """
    Reads the downloaded Kaggle dataset and splits it into the two
    specific CSV files needed for our project.
    """
    print(f"--- Formatting Kaggle Dataset: {KAGGLE_INPUT_FILE} ---")

    try:
        df = pd.read_csv(KAGGLE_INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{KAGGLE_INPUT_FILE}'.")
        print("Please download the dataset from Kaggle and place it in your 'data/' folder.")
        return

    # --- 1. Create phishing_text_dataset.csv ---
    print("[*] Creating the NLP text dataset...")
    # Select the 'Text' and 'Label' columns and rename them to match our project
    df_nlp = df[['Text', 'Label']].copy()
    df_nlp.rename(columns={'Text': 'text', 'Label': 'label'}, inplace=True)
    # Remove any rows where the text might be missing
    df_nlp.dropna(subset=['text'], inplace=True)
    df_nlp.to_csv(NLP_OUTPUT_FILE, index=False)
    print(f"✅ NLP dataset created at '{NLP_OUTPUT_FILE}' with {len(df_nlp)} rows.")

    # --- 2. Create structured_features_dataset.csv ---
    print("[*] Creating the structured features dataset...")
    # NOTE: We can only create features derivable from the URL itself, since we are not
    # making live web requests. Header-based features will be missing.
    structured_records = []
    for index, row in df.iterrows():
        url = row['URL']
        label = row['Label']
        try:
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            if not hostname:
                continue

            features = {}
            features['uses_https'] = 1 if url.startswith("https://") else 0
            domain_part = hostname.split('.')[-2] if '.' in hostname else hostname
            features['domain_digit_count'] = sum(c.isdigit() for c in domain_part)
            features['url_length'] = len(url)
            features['hostname_length'] = len(hostname)
            features['path_level'] = url.count('/')
            # We add placeholders for header features since we can't get them from a static file
            features['csp_present'] = 0
            features['hsts_present'] = 0
            features['xfo_present'] = 0
            features['is_phishing'] = label
            structured_records.append(features)
        except Exception:
            continue # Skip any malformed URLs

    df_structured = pd.DataFrame(structured_records)
    df_structured.to_csv(STRUCTURED_OUTPUT_FILE, index=False)
    print(f"✅ Structured features dataset created at '{STRUCTURED_OUTPUT_FILE}' with {len(df_structured)} rows.")
    print("\n--- You are now ready to train your models! ---")


if __name__ == "__main__":
    format_data()