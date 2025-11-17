# create_sample_list.py

import pandas as pd

# --- Configuration ---
# This is your big, original file
FULL_LIST_FILE = 'data/url_list_for_training_FULL.csv'
# This is the new, smaller file we will create
SAMPLE_OUTPUT_FILE = 'data/url_list_for_training.csv'
# We'll take 600 phishing and 600 safe URLs for a balanced 1200 total
NUM_SAMPLES_PER_CLASS = 1000

def create_sample_list():
    """
    Reads the full URL list and creates a smaller, balanced sample for
    faster testing and development.
    """
    print("--- Creating a smaller sample URL list (approx. 1200 URLs) ---")
    try:
        # 1. Read the full dataset
        df_full = pd.read_csv(FULL_LIST_FILE)
        print(f"[*] Full dataset loaded with {len(df_full)} URLs.")
    except FileNotFoundError:
        print(f"❌ ERROR: The full dataset file '{FULL_LIST_FILE}' was not found.")
        print("Please make sure you have renamed your large URL list to this name.")
        return

    # 2. Separate the data into phishing and safe URLs
    df_malicious = df_full[df_full['label'] == 1]
    df_safe = df_full[df_full['label'] == 0]

    print(f"    -> Found {len(df_malicious)} malicious and {len(df_safe)} safe URLs in the full list.")

    # 3. Take a random sample from each category
    # Using .min() prevents errors if you have fewer than 600 of one type
    malicious_sample_size = min(NUM_SAMPLES_PER_CLASS, len(df_malicious))
    safe_sample_size = min(NUM_SAMPLES_PER_CLASS, len(df_safe))

    malicious_sample = df_malicious.sample(n=malicious_sample_size, random_state=42)
    safe_sample = df_safe.sample(n=safe_sample_size, random_state=42)

    print(f"[*] Taking a sample of {len(malicious_sample)} malicious and {len(safe_sample)} safe URLs.")

    # 4. Combine the samples into one dataframe
    final_df = pd.concat([malicious_sample, safe_sample], ignore_index=True)

    # 5. Shuffle the final combined list to mix them up
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 6. Save the new, smaller list
    final_df.to_csv(SAMPLE_OUTPUT_FILE, index=False)

    print(f"\n✅ Success! Sample list created at '{SAMPLE_OUTPUT_FILE}' with {len(final_df)} total URLs.")
    print("You can now run 'python scripts/create_datasets.py' with this smaller list for much faster processing.")

if __name__ == "__main__":
    create_sample_list()