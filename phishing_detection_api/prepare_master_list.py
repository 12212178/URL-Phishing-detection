# prepare_master_list.py (Corrected Version)

import pandas as pd

# --- Configuration ---
NUM_SAFE_URLS = 20000
# Make sure your filenames in the 'data' folder match these exactly
URLHAUS_FILE = 'data/urlhaus_data.csv'
TRANCO_FILE = 'data/tranco_YL7PG.csv'
OUTPUT_FILE = 'data/url_list_for_training.csv'

def prepare_master_list():
    """
    Combines malicious URLs from URLhaus and safe URLs from Tranco
    into a single, labeled CSV file ready for the next step.
    """
    print("--- Starting Master URL List Creation ---")

    # --- 1. Process Malicious URLs (URLhaus) ---
    print(f"[*] Processing malicious URLs from {URLHAUS_FILE}...")
    try:
        # THIS IS THE FIX: We explicitly define the column names based on the file's structure.
        column_names = [
            'id', 'dateadded', 'url', 'url_status', 'last_online',
            'threat', 'tags', 'urlhaus_link', 'reporter'
        ]
        df_malicious = pd.read_csv(
            URLHAUS_FILE,
            comment='#',      # Skip comment lines at the top of the file
            header=None,      # Tell pandas there is NO header row to read
            names=column_names # Provide the correct column names ourselves
        )

        # Now, we can reliably select the correct 'url' column
        df_malicious = df_malicious[['url']]
        df_malicious['label'] = 1 # Assign label 1 for phishing
        print(f"    -> Found {len(df_malicious)} malicious URLs.")

    except Exception as e:
        print(f"❌ ERROR: Could not process {URLHAUS_FILE}. Error: {e}")
        return

    # --- 2. Process Safe URLs (Tranco) ---
    print(f"[*] Processing safe URLs from {TRANCO_FILE}...")
    try:
        df_safe = pd.read_csv(TRANCO_FILE, names=['rank', 'domain'])
        df_safe = df_safe.head(NUM_SAFE_URLS)
        df_safe['url'] = 'https://www.' + df_safe['domain']
        df_safe['label'] = 0 # Assign label 0 for safe
        df_safe = df_safe[['url', 'label']]
        print(f"    -> Selected top {len(df_safe)} safe URLs.")
    except Exception as e:
        print(f"❌ ERROR: Could not process {TRANCO_FILE}. Error: {e}")
        return

    # --- 3. Combine, Shuffle, and Save ---
    print("[*] Combining and shuffling datasets...")
    final_df = pd.concat([df_malicious, df_safe], ignore_index=True)
    # Remove any rows where the URL might be invalid or missing
    final_df.dropna(subset=['url'], inplace=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Success! Master list created at '{OUTPUT_FILE}' with {len(final_df)} total URLs.")

if __name__ == "__main__":
    prepare_master_list()