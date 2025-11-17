import pandas as pd
import joblib
import keras_tuner as kt  # <-- NEW: Import KerasTuner
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np

# --- Configuration ---
MAX_WORDS = 20000
MAX_LEN = 200

def build_model(hp):
    """
    This function builds the Keras model and defines the
    hyperparameter search space for KerasTuner.
    """
    # --- Define Search Space ---
    
    # 1. Embedding layer
    hp_embed_dim = hp.Int(name="embedding_dim", min_value=64, max_value=256, step=64)
    
    # 2. Conv1D layer
    hp_filters = hp.Int(name="filters", min_value=32, max_value=128, step=32)
    hp_kernel_size = hp.Choice(name="kernel_size", values=[3, 5])

    # 3. LSTM layer
    hp_lstm_units = hp.Int(name="lstm_units", min_value=32, max_value=128, step=32)

    # 4. Dropout rates
    hp_dropout = hp.Float(name="dropout", min_value=0.2, max_value=0.5, step=0.1)
    
    # 5. Learning rate
    hp_learning_rate = hp.Float(name="learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    # --- Build the Model Architecture ---
    input_layer = Input(shape=(MAX_LEN,))
    
    embedding_layer = Embedding(
        input_dim=MAX_WORDS, 
        output_dim=hp_embed_dim
    )(input_layer)
    
    # CNN layer extracts key local features
    conv_layer = Conv1D(
        filters=hp_filters, 
        kernel_size=hp_kernel_size, 
        activation='relu'
    )(embedding_layer)
    
    # NEW: Pooling layer to reduce dimensionality and summarize features
    pool_layer = MaxPooling1D(pool_size=2)(conv_layer)
    
    # LSTM layer processes the sequence of features
    lstm_layer = LSTM(
        units=hp_lstm_units, 
        dropout=hp_dropout, 
        recurrent_dropout=hp_dropout  # Regularizes the LSTM cell
    )(pool_layer)
    
    # Dense layers for final classification
    dense_layer = Dense(64, activation='relu')(lstm_layer)
    dropout_layer = Dropout(hp_dropout)(dense_layer)
    output_layer = Dense(1, activation='sigmoid')(dropout_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with the tunable learning rate
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


def train_tuned_cnn_lstm():
    """
    Loads data, uses KerasTuner to find the best hyperparameters,
    and then trains a final model.
    """
    print("--- Starting Tuned CNN-LSTM Model Training ---")

    # --- 1. Load Your Custom Dataset ---
    try:
        df = pd.read_csv('../data/phishing_text_dataset.csv')
    except FileNotFoundError:
        print("❌ ERROR: 'data/phishing_text_dataset.csv' not found. Please run 'create_datasets.py' first.")
        return
    
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    X = df['text']
    y = df['label']

    # --- 2. Tokenize and Pad Text Data ---
    print("[*] Tokenizing and padding text data...")
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(sequences, maxlen=MAX_LEN)
    
    # Save tokenizer *before* splitting data (it's trained on all text)
    joblib.dump(tokenizer, '../ml_models/cnn_lstm_tokenizer.joblib')
    print("[*] Tokenizer saved.")

    # --- 3. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 4. Run Hyperparameter Search ---
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',  # Goal is to maximize validation accuracy
        max_trials=5,             # Try 5 different combinations
        executions_per_trial=1,    # Train each combination once
        directory='keras_tuner_logs',
        project_name='cnn_lstm_phishing'
    )

    # We need to use EarlyStopping during the search
    search_callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

    print("[*] Starting hyperparameter search...")
    tuner.search(
        X_train,
        y_train,
        epochs=5,  # Set a high number, EarlyStopping will handle it
        validation_data=(X_test, y_test),
        callbacks=search_callbacks,
        batch_size=64
    )

    print("[*] Search complete.")
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\n--- Best Hyperparameters Found ---")
    print(f"Embedding Dim: {best_hps.get('embedding_dim')}")
    print(f"Filters: {best_hps.get('filters')}")
    print(f"Kernel Size: {best_hps.get('kernel_size')}")
    print(f"LSTM Units: {best_hps.get('lstm_units')}")
    print(f"Dropout: {best_hps.get('dropout')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")

    # --- 5. Train the Final, Best Model ---
    print("\n[*] Building and training the final optimized model...")
    final_model = tuner.hypermodel.build(best_hps)
    
    # Use EarlyStopping again for the final training
    final_callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

    final_model.fit(
        X_train,
        y_train,
        epochs=50,  # Train for longer, EarlyStopping will find the best spot
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=final_callbacks
    )

    # --- 6. Evaluate the Final Model ---
    print("\n--- Final Model Evaluation on Test Set ---")
    y_pred_proba = final_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype("int32")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Phishing (1)']))

    # --- 7. Save the Final Model ---
    final_model.save('../ml_models/cnn_lstm_model.h5')
    print("✅ Final optimized CNN-LSTM model saved to 'ml_models/cnn_lstm_model.h5'")

if __name__ == "__main__":
    train_tuned_cnn_lstm()