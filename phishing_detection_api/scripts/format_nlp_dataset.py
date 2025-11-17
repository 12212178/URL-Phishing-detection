# format_nlp_dataset.py (Corrected Version)

import pandas as pd

# --- Configuration ---
KAGGLE_INPUT_FILE = 'data/Phishing_Email.csv'
NLP_OUTPUT_FILE = 'data/phishing_text_dataset.csv'

def format_nlp_data():
    """
    Reads the downloaded Phishing Email dataset and formats it into the
    'phishing_text_dataset.csv' file needed for our NLP models.
    """
    print(f"--- Formatting Kaggle Dataset for NLP: {KAGGLE_INPUT_FILE} ---")

    try:
        # --- THIS IS THE FIX ---
        # We switch to the 'python' engine, which is more flexible,
        # and tell it to skip any rows that cause parsing errors.
        df = pd.read_csv(
            KAGGLE_INPUT_FILE,
            engine='python',
            on_bad_lines='skip'
        )
        # --- END OF FIX ---

    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{KAGGLE_INPUT_FILE}'.")
        print("Please download the Phishing Email dataset and place it in your 'data/' folder.")
        return

    # --- 1. Select and Rename Columns ---
    if 'Email Text' not in df.columns or 'Email Type' not in df.columns:
        print("❌ ERROR: Required columns ('Email Text', 'Email Type') not found.")
        return

    df_nlp = df[['Email Text', 'Email Type']].copy()
    df_nlp.rename(columns={'Email Text': 'text', 'Email Type': 'label'}, inplace=True)

    # --- 2. Convert Text Labels to Numbers (0 and 1) ---
    label_map = {'Phishing Email': 1, 'Safe Email': 0}
    df_nlp['label'] = df_nlp['label'].map(label_map)

    # --- 3. Clean Up Data ---
    df_nlp.dropna(subset=['text', 'label'], inplace=True)
    df_nlp['label'] = df_nlp['label'].astype(int)

    # --- 4. Save the Final Dataset ---
    df_nlp.to_csv(NLP_OUTPUT_FILE, index=False)
    print(f"✅ NLP dataset created at '{NLP_OUTPUT_FILE}' with {len(df_nlp)} rows.")
    print("\nYou are now ready to train your NLP models (BERT and CNN-LSTM).")

if __name__ == "__main__":
    format_nlp_data()