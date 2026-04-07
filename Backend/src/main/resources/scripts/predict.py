#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict cryptographic algorithm from hex-encoded ciphertext.

Called by Spring Boot via ProcessBuilder:
    python3 predict.py <HEX_DATA>

Prints the predicted algorithm name (e.g. "AES", "SHA256") to stdout.
Uses the same feature extraction as train_model.py so features match.
"""

import sys
import os
import pickle
import json
import numpy as np

# Add script directory so sibling modules (FeatureExtraction) can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from FeatureExtraction import extract_features_from_hex

MAX_STREAKS = 10
CONFIDENCE_THRESHOLD = float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', '0.75'))
MODEL_VERSION = os.getenv('PREDICTION_MODEL_VERSION', 'hybrid-rf-lr-v2')

try:
    # Ensure the argument is provided
    if len(sys.argv) < 2:
        print("Error: No HEX input provided.")
        sys.exit(1)

    input_hex = sys.argv[1]

    # Load hybrid model (dict with rf_model, lr_model, encoder)
    model_path = os.path.join(script_dir, 'model.pickle')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Load label map (maps numeric label → algorithm name)
    label_map_path = os.path.join(script_dir, 'label_map.pickle')
    label_map = None
    if os.path.exists(label_map_path):
        with open(label_map_path, 'rb') as f:
            label_map = pickle.load(f)

    # Extract features (same pipeline as training)
    features = extract_features_from_hex(input_hex, MAX_STREAKS)
    features_array = np.array(features).reshape(1, -1)
    confidence = None

    # Hybrid prediction: RF → one-hot encode → combine → scale → LR
    if isinstance(model_data, dict) and 'rf_model' in model_data:
        rf_model = model_data['rf_model']
        lr_model = model_data['lr_model']
        encoder = model_data['encoder']
        scaler = model_data.get('scaler')

        rf_pred = rf_model.predict(features_array)
        rf_encoded = encoder.transform(rf_pred.reshape(-1, 1))
        combined = np.hstack((features_array, rf_encoded))
        if scaler is not None:
            combined = scaler.transform(combined)

        if hasattr(lr_model, 'predict_proba'):
            probabilities = lr_model.predict_proba(combined)[0]
            classes = lr_model.classes_
            best_idx = int(np.argmax(probabilities))
            prediction = classes[best_idx]
            confidence = float(probabilities[best_idx])
        else:
            prediction = lr_model.predict(combined)[0]
    else:
        # Fallback: plain model (backward compatible)
        if hasattr(model_data, 'predict_proba'):
            probabilities = model_data.predict_proba(features_array)[0]
            classes = model_data.classes_
            best_idx = int(np.argmax(probabilities))
            prediction = classes[best_idx]
            confidence = float(probabilities[best_idx])
        else:
            prediction = model_data.predict(features_array)[0]

    if isinstance(prediction, np.integer):
        prediction = int(prediction)

    # Map numeric label to algorithm name
    if label_map and prediction in label_map:
        result = label_map[prediction]
    else:
        result = str(prediction)

    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        result = 'Unknown'

    payload = {
        "predicted_algorithm": result,
        "confidence": confidence,
        "model_version": MODEL_VERSION,
        "source": "ml_fallback",
    }
    print(json.dumps(payload))  # Spring Boot reads this from stdout

except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)
