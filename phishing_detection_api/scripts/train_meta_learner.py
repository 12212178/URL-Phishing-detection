import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import tqdm
from scipy.stats import loguniform  # For a better search space for 'C'

def train_meta_learner():
    """
    Loads predictions from all base models on a test set, then trains
    and evaluates a final meta-learner on them.
    """
    print("--- Starting Meta-Learner Training Process ---")

    # --- 1. Load Datasets ---
    try:
        df_structured = pd.read_csv('data/dataset_B_05_2020.csv')
        df_text = pd.read_csv('data/phishing_text_dataset.csv')
    except FileNotFoundError:
        print("❌ ERROR: One or more datasets not found. Please ensure both datasets are present.")
        return

    # --- 2. Align and Split Data Consistently ---
    min_len = min(len(df_structured), len(df_text))
    df_structured = df_structured.head(min_len)
    df_text = df_text.head(min_len)

    TARGET_COLUMN = 'status'
    # Assuming 'status' is the correct column. If it's 'phishing', change this.
    try:
        df_structured[TARGET_COLUMN] = df_structured[TARGET_COLUMN].replace({'legitimate': 0, 'phishing': 1})
        X_structured = df_structured.drop(columns=[TARGET_COLUMN, 'url'])
        y = df_structured[TARGET_COLUMN]
    except KeyError:
        # Fallback for the other dataset's schema
        TARGET_COLUMN = 'phishing'
        print(f"[*] Warning: 'status' column not found, falling back to '{TARGET_COLUMN}'.")
        X_structured = df_structured.drop(columns=[TARGET_COLUMN])
        y = df_structured[TARGET_COLUMN]
    
    # Split data to get a consistent "base" test set (the 20% the base models never saw)
    # We don't need the _train split, as base models are already trained
    _, X_base_test, _, y_base_test = train_test_split(
        X_structured, y, test_size=0.2, random_state=42, stratify=y
    )
    # Get the corresponding text data for this test set
    text_base_test = df_text.loc[y_base_test.index]['text'].astype(str).tolist()

    # --- 3. Load All Trained Base Models ---
    print("[*] Loading all base models...")
    try:
        # Use the *tuned* models if they exist, otherwise fall back
        try:
            xgb_model = joblib.load('ml_models/xgboost_tuned_model.joblib')
            print("    -> Loaded tuned XGBoost model.")
        except FileNotFoundError:
            xgb_model = joblib.load('ml_models/xgboost_model.joblib')
            print("    -> Loaded non-tuned XGBoost model.")
            
        nlp_classifier = pipeline("text-classification", model="ml_models/my_finetuned_phishing_model", device=-1)
        cnn_model = load_model('ml_models/cnn_lstm_model.h5')
        cnn_tokenizer = joblib.load('ml_models/cnn_lstm_tokenizer.joblib')
    except Exception as e:
        print(f"❌ ERROR: Could not load one or more base models. Please train them first. Error: {e}")
        return

    # --- 4. Generate Predictions from Base Models on the *Entire* Test Set ---
    print("[*] Generating predictions from base models (this may take a moment)...")
    
    xgb_preds = xgb_model.predict_proba(X_base_test)[:, 1]

    bert_preds = []
    for text in tqdm(text_base_test, desc="BERT Predictions"):
        result = nlp_classifier(text, truncation=True, max_length=512)[0]
        bert_preds.append(result['score'] if result['label'] == 'LABEL_1' else 1 - result['score'])
    bert_preds = np.array(bert_preds)

    sequences = cnn_tokenizer.texts_to_sequences(text_base_test)
    padded = pad_sequences(sequences, maxlen=200)
    cnn_preds = cnn_model.predict(padded, verbose=0).flatten()

    # Create the full meta-feature dataset
    meta_features_full = pd.DataFrame({
        'xgb_pred': xgb_preds,
        'bert_pred': bert_preds,
        'cnn_pred': cnn_preds
    })

    # --- 5. [FIX] Split Meta-Features for *Proper* Training and Evaluation ---
    # This is the critical step to prevent data leakage.
    # We split the predictions (meta-features) and their labels (y_base_test)
    X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
        meta_features_full, 
        y_base_test, 
        test_size=0.25, # e.g., 25% of the 20% -> 5% of total data for final test
        random_state=42, 
        stratify=y_base_test
    )
    
    print(f"[*] Meta-features split: {len(X_meta_train)} for training, {len(X_meta_test)} for testing.")

    # --- 6. Define Search Space and Tune the Meta-Learner ---
    meta_learner = LogisticRegression()
    
    param_dist = {
        'C': loguniform(0.01, 100),  # Inverse of regularization strength
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']      # Good solver for this small-scale problem
    }

    print("[*] Tuning the final meta-learner...")
    tuner = RandomizedSearchCV(
        estimator=meta_learner,
        param_distributions=param_dist,
        n_iter=20,  # Try 20 combinations
        cv=5,       # 5-fold cross-validation on the meta-train data
        scoring='f1', # Optimize for F1-score
        n_jobs=-1,
        random_state=42
    )
    
    tuner.fit(X_meta_train, y_meta_train)
    
    print("[*] Tuning complete.")
    print(f"Best Meta-Learner F1-Score: {tuner.best_score_:.2%}")
    print(f"Best Meta-Learner Params: {tuner.best_params_}")
    
    # Get the best, fully-tuned model
    best_meta_learner = tuner.best_estimator_

    # --- 7. Evaluate the Final Ensemble on the *Unseen* Meta-Test Set ---
    print("\n--- Final Ensemble Performance (on unseen meta-test data) ---")
    final_predictions = best_meta_learner.predict(X_meta_test)
    
    print("Classification Report for the Complete Ensemble:")
    print(classification_report(y_meta_test, final_predictions, target_names=['legitimate (0)', 'phishing (1)']))

    # --- 8. Save the Trained Meta-Learner ---
    joblib.dump(best_meta_learner, 'ml_models/meta_learner.joblib')
    print("✅ Tuned meta-learner trained and saved successfully to 'ml_models/meta_learner.joblib'")

if __name__ == "__main__":
    train_meta_learner()