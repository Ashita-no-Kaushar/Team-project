#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the cryptographic algorithm identification model (Hybrid approach).

Uses a two-stage model from model.py:
  Stage 1 — Random Forest produces predictions
  Stage 2 — Logistic Regression refines using original features + RF one-hot output

Pipeline:
  1. Generate synthetic cipher & hash samples  (Corpus.py)
  2. Extract features from ciphertext only     (FeatureExtraction.py)
  3. Train hybrid RF + LR model
  4. Evaluate on a held-out test set
  5. Save  model.pickle  +  label_map.pickle

Usage:
    python3 train_model.py                     # default 500 samples / algo
    python3 train_model.py --samples 1000      # custom count
"""

import os
import sys
import pickle
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add script directory to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from Corpus import (
    generate_cipher_samples,
    generate_hash_samples,
    ALGORITHM_LABELS,
    LABEL_NAMES,
)

MAX_STREAKS = 10
PLAINTEXT_SIZES = [16, 32, 64, 128, 256, 512, 1024]


def main():
    parser = argparse.ArgumentParser(
        description="Train crypto-algorithm identification model (hybrid RF+LR)"
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Number of samples to generate per algorithm (default: 500)"
    )
    args = parser.parse_args()
    n_samples = args.samples

    print("=" * 60)
    print(" Cryptographic Algorithm Identification — Hybrid Model Training")
    print("=" * 60)

    # ── Step 1 & 2: generate data + extract features ─────────────────────
    all_features = []
    all_labels = []

    print(f"\n[1/5] Generating cipher samples ({n_samples} per algorithm)...")
    cf, cl = generate_cipher_samples(n_samples, PLAINTEXT_SIZES, MAX_STREAKS)
    all_features.extend(cf)
    all_labels.extend(cl)

    print(f"\n[2/5] Generating hash samples ({n_samples} per algorithm)...")
    hf, hl = generate_hash_samples(n_samples, PLAINTEXT_SIZES, MAX_STREAKS)
    all_features.extend(hf)
    all_labels.extend(hl)

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\n  Total samples       : {len(X)}")
    print(f"  Feature vector size : {X.shape[1]}")
    print(f"  Algorithms          : {len(set(y))}")
    for label_id in sorted(set(y)):
        count = int(np.sum(y == label_id))
        print(f"    {LABEL_NAMES[label_id]:>10s} : {count}")

    # ── Step 3: train/test split ─────────────────────────────────────────
    print("\n[3/5] Splitting data (75% train / 25% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )

    # ── Step 4: train hybrid model (RF → one-hot → LR) ──────────────────
    print("\n[4/5] Training Hybrid model (Random Forest + Logistic Regression)...")

    # Stage 1: Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    rf_train_pred = rf_model.predict(X_train)
    rf_train_acc = accuracy_score(y_train, rf_train_pred)
    print(f"  RF train accuracy: {rf_train_acc:.4f}")

    # One-hot encode RF predictions
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    rf_encoded_train = encoder.fit_transform(rf_train_pred.reshape(-1, 1))

    # Combine original features + RF one-hot predictions
    X_train_combined = np.hstack((X_train, rf_encoded_train))

    # Scale combined features for LR convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)

    # Stage 2: Logistic Regression on scaled combined features
    lr_model = LogisticRegression(max_iter=5000, random_state=42, solver='lbfgs')
    lr_model.fit(X_train_scaled, y_train)

    # ── Evaluate on test set ─────────────────────────────────────────────
    rf_test_pred = rf_model.predict(X_test)
    rf_test_encoded = encoder.transform(rf_test_pred.reshape(-1, 1))
    X_test_combined = np.hstack((X_test, rf_test_encoded))
    X_test_scaled = scaler.transform(X_test_combined)

    # RF-only accuracy
    rf_acc = accuracy_score(y_test, rf_test_pred)
    print(f"\n  RF-only Test Accuracy : {rf_acc:.4f}")

    # Hybrid model accuracy
    final_predictions = lr_model.predict(X_test_scaled)
    hybrid_acc = accuracy_score(y_test, final_predictions)
    print(f"  Hybrid  Test Accuracy : {hybrid_acc:.4f}")

    target_names = [LABEL_NAMES[i] for i in sorted(set(y))]
    print("\n  Classification Report (Hybrid Model):")
    print(classification_report(y_test, final_predictions, target_names=target_names))

    # ── Step 5: save artefacts ───────────────────────────────────────────
    print("[5/5] Saving model artefacts...")

    model_path = os.path.join(_SCRIPT_DIR, "model.pickle")
    label_map_path = os.path.join(_SCRIPT_DIR, "label_map.pickle")

    # Save all three components needed for prediction
    hybrid_model = {
        'rf_model': rf_model,
        'lr_model': lr_model,
        'encoder': encoder,
        'scaler': scaler,
    }

    with open(model_path, "wb") as f:
        pickle.dump(hybrid_model, f)
    print(f"  Hybrid model saved to : {model_path}")

    with open(label_map_path, "wb") as f:
        pickle.dump(LABEL_NAMES, f)
    print(f"  Label map saved to    : {label_map_path}")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
