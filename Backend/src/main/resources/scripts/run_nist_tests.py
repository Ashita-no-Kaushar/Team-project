#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Extractor — runs NIST SP 800-22 statistical tests on encrypted data files.

This is a supplementary feature extractor that runs NIST randomness tests.
It requires the sp800_22_tests Python package to be available.

Usage:
    python3 run_nist_tests.py [--data-dir <path>] [--nist-dir <path>] [--output-dir <path>]

Defaults:
    --data-dir   : ./encrypted_data
    --nist-dir   : ./sp800_22_tests     (path to the NIST test suite)
    --output-dir : ./nist_test_results
"""

import os
import sys
import argparse
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_nist_tests(file_path, sp800_22_path, output_dir):
    """Run the NIST SP 800-22 test suite on a single binary file."""
    output_file = os.path.join(
        output_dir,
        os.path.basename(file_path).replace('.bin', '_nist_results.txt')
    )

    command = [sys.executable, sp800_22_path, file_path]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error on {file_path}: {result.stderr.strip()}")
        else:
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            print(f"  Done: {os.path.basename(file_path)} -> {output_file}")
    except Exception as e:
        print(f"  Exception: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run NIST SP 800-22 tests on encrypted data")
    parser.add_argument("--data-dir", default=os.path.join(_SCRIPT_DIR, "encrypted_data"),
                        help="Directory containing encrypted .bin files")
    parser.add_argument("--nist-dir", default=os.path.join(_SCRIPT_DIR, "sp800_22_tests"),
                        help="Path to the sp800_22_tests directory")
    parser.add_argument("--output-dir", default=os.path.join(_SCRIPT_DIR, "nist_test_results"),
                        help="Directory to store NIST test results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sp800_22_path = os.path.join(args.nist_dir, "sp800_22_tests.py")
    if not os.path.exists(sp800_22_path):
        print(f"ERROR: sp800_22_tests.py not found at: {sp800_22_path}")
        print("Please download the NIST SP 800-22 test suite and place it in the --nist-dir path.")
        sys.exit(1)
    else:
        print(f"Found sp800_22_tests.py at: {sp800_22_path}")

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    files = [f for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f))]
    print(f"Found {len(files)} files in {args.data_dir}\n")

    for filename in files:
        file_path = os.path.join(args.data_dir, filename)
        print(f"Running NIST tests on {filename}...")
        run_nist_tests(file_path, sp800_22_path, args.output_dir)

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
