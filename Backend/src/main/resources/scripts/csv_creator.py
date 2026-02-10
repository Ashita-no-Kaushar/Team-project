#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Pre-processor — parses NIST SP 800-22 test results into a CSV dataset.

Reads result text files produced by run_nist_tests.py and extracts statistical
p-values as features, producing a labelled CSV suitable for model training.

Usage:
    python3 csv_creator.py [--results-dir <path>] [--output <file>]

Defaults:
    --results-dir : ./nist_test_results
    --output      : ./nist_feature_dataset.csv
"""

import re
import os
import sys
import argparse
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_features_from_file(file_path):
    """Extract NIST test p-values and statistics from a result file."""
    features = {}
    with open(file_path, 'r') as f:
        content = f.read()

    # Monobit test
    match = re.search(
        r'TEST: monobit_test.*?Ones count\s*=\s*(\d+).*?Zeroes count\s*=\s*(\d+).*?P=([0-9.]+)',
        content, re.DOTALL
    )
    if match:
        features['monobit_test_ones_count'] = int(match.group(1))
        features['monobit_test_zeroes_count'] = int(match.group(2))
        features['monobit_test_p'] = float(match.group(3))
    else:
        features['monobit_test_ones_count'] = None
        features['monobit_test_zeroes_count'] = None
        features['monobit_test_p'] = None

    # Frequency within block test
    match = re.search(r'TEST: frequency_within_block_test.*?P=([0-9.]+)', content, re.DOTALL)
    features['frequency_within_block_test_p'] = float(match.group(1)) if match else None

    # Runs test
    match = re.search(r'TEST: runs_test.*?P=([0-9.]+)', content, re.DOTALL)
    features['runs_test_p'] = float(match.group(1)) if match else None

    # Longest run of ones in a block test
    match = re.search(r'TEST: longest_run_ones_in_a_block_test.*?P=([0-9.]+)', content, re.DOTALL)
    features['longest_run_ones_in_a_block_test_p'] = float(match.group(1)) if match else None

    # Binary matrix rank test
    match = re.search(r'TEST: binary_matrix_rank_test.*?P=([0-9.]+)', content, re.DOTALL)
    features['binary_matrix_rank_test_p'] = float(match.group(1)) if match else None

    # DFT test
    match = re.search(r'TEST: dft_test.*?P=([0-9.]+)', content, re.DOTALL)
    features['dft_test_p'] = float(match.group(1)) if match else None

    # Approximate entropy test
    match = re.search(r'TEST: approximate_entropy_test.*?P=([0-9.]+)', content, re.DOTALL)
    features['approximate_entropy_test_p'] = float(match.group(1)) if match else None

    # Serial test
    match = re.search(r'TEST: serial_test.*?P=([0-9.]+)', content, re.DOTALL)
    features['serial_test_p'] = float(match.group(1)) if match else None

    # Cumulative sums test
    match = re.search(r'TEST: cumulative_sums_test.*?P=([0-9.]+)', content, re.DOTALL)
    features['cumulative_sums_test_p'] = float(match.group(1)) if match else None

    # Non-overlapping template matching test
    match = re.search(r'TEST: non_overlapping_template_matching_test.*?P=([0-9.]+)', content, re.DOTALL)
    features['non_overlapping_template_matching_test_p'] = float(match.group(1)) if match else None

    return features


def main():
    parser = argparse.ArgumentParser(description="Parse NIST test results into CSV")
    parser.add_argument("--results-dir", default=os.path.join(_SCRIPT_DIR, "nist_test_results"),
                        help="Directory containing NIST result files")
    parser.add_argument("--output", default=os.path.join(_SCRIPT_DIR, "nist_feature_dataset.csv"),
                        help="Output CSV file path")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)

    dataset = []
    files = sorted(f for f in os.listdir(args.results_dir) if f.endswith("_nist_results.txt"))
    print(f"Found {len(files)} result files in {args.results_dir}")

    for filename in files:
        file_path = os.path.join(args.results_dir, filename)
        features = extract_features_from_file(file_path)

        # Extract the label from filename (e.g., "AES_8KB_0_nist_results.txt" -> "AES")
        label = filename.split('_')[0]
        features['label'] = label
        dataset.append(features)

    df = pd.DataFrame(dataset)
    df.to_csv(args.output, index=False)
    print(f"Dataset saved to {args.output} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
