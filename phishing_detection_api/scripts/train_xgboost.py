import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint, uniform  # For defining search ranges
import time  # To time the tuning process

# --- Configuration ---
INPUT_FILE = '../data/dataset_B_05_2020.csv'
TARGET_COLUMN = 'status'
# Define a new name for the tuned model
MODEL_OUTPUT_FILE = '../ml_models/xgboost_tuned_model.joblib'

def tune_and_train_xgboost():
    """
    Loads the dataset, uses RandomizedSearchCV to find the best
    hyperparameters for XGBoost, trains on the full data, evaluates,
    and saves the final optimized model.
    """
    print("--- Tuning and Training Optimized XGBoost ---")
    start_time = time.time()

    # --- 1. Load Dataset ---
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ ERROR: '{INPUT_FILE}' not found. Please place the downloaded dataset in your 'data/' folder.")
        return
    except KeyError:
        print(f"❌ ERROR: Target column '{TARGET_COLUMN}' not found in the dataset.")
        print("Please open the CSV and verify the name of the label column.")
        return

    # --- 2. Prepare Data and Split ---
    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({'legitimate': 0, 'phishing': 1})
    X = df.drop(columns=[TARGET_COLUMN, 'url'])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- 3. Define Model and Hyperparameter Search Space ---
    
    # Initialize the base model
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss"
    )

    # Define the parameter distributions to search
    # This is where you set the ranges for the tuner
    param_dist = {
        'n_estimators': randint(200, 1000),      # Number of trees
        'learning_rate': uniform(0.01, 0.2),     # Step size
        'max_depth': randint(3, 10),           # Max tree depth
        'subsample': uniform(0.6, 0.4),        # % of samples per tree (range 0.6 to 1.0)
        'colsample_bytree': uniform(0.6, 0.4)  # % of features per tree (range 0.6 to 1.0)
    }

    # --- 4. Initialize and Run the Randomized Search ---
    print("[*] Starting hyperparameter tuning with RandomizedSearchCV...")
    
    # n_iter=50 means it will try 50 different random combinations
    # cv=5 means 5-fold cross-validation for each combination
    # n_jobs=-1 uses all available CPU cores to speed up the search
    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
        random_state=42  # For reproducible results
    )

    # Fit the tuner to find the best parameters
    tuner.fit(X_train, y_train)

    print("[*] Tuning complete.")
    print(f"[*] Total tuning time: {time.time() - start_time:.2f} seconds")

    # --- 5. Get Best Model and Evaluate ---
    print("\n--- Model Evaluation ---")
    
    # Get the best parameters and the best score from the search
    print(f"Best Cross-Validation Accuracy: {tuner.best_score_:.2%}")
    print("Best Hyperparameters Found:")
    print(tuner.best_params_)

    # The 'tuner' object automatically refits the best model on the *entire* X_train
    # This is the final, optimized model
    best_model = tuner.best_estimator_

    # Evaluate the best model on the hold-out test set
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest Set Accuracy: {accuracy:.2%}")
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Safe (0)', 'Phishing (1)']))

    # --- 6. Save the Tuned Model ---
    joblib.dump(best_model, MODEL_OUTPUT_FILE)
    print(f"✅ Tuned XGBoost model saved to '{MODEL_OUTPUT_FILE}'")

if __name__ == "__main__":
    tune_and_train_xgboost()