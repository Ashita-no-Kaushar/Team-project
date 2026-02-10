#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extraction functions for cryptographic algorithm identification.

Extracts both byte-level and bit-level statistical features from ciphertext/hash data.
These features are used for training and prediction.
"""

import numpy as np
from math import log
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Binary-string level features (from original FeatureExtraction.py, modernised)
# ---------------------------------------------------------------------------

def bytes_to_binary_string(data_bytes):
    """Convert bytes to a binary string of '0' and '1' characters."""
    return ''.join(format(b, '08b') for b in data_bytes)


def get_streaks(binary_string, max_len_streaks=10):
    """Count consecutive runs (streaks) of identical bits.

    Returns a list of length (max_len_streaks + 1):
      - [0..max_len_streaks-1] = normalised count of streaks of that length
      - [-1] = total number of streaks
    """
    if not binary_string:
        return [0.0] * (max_len_streaks + 1)

    runs = []
    current_run_length = 1
    for i in range(1, len(binary_string)):
        if binary_string[i] == binary_string[i - 1]:
            current_run_length += 1
        else:
            runs.append(current_run_length)
            current_run_length = 1
    runs.append(current_run_length)  # last run

    total = len(binary_string)
    rn = [0.0] * max_len_streaks
    for r in runs:
        if r <= max_len_streaks:
            rn[r - 1] += 1
    for i in range(len(rn)):
        rn[i] /= total

    return rn + [float(len(runs))]


def get_frequencies(binary_string):
    """Get normalised frequency of '0' and '1' bits."""
    total = len(binary_string)
    if total == 0:
        return [0.0, 0.0]
    return [float(binary_string.count('0')) / total,
            float(binary_string.count('1')) / total]


def get_entropy(frequencies):
    """Calculate binary entropy from [freq_0, freq_1]."""
    if frequencies[0] <= 0 or frequencies[1] <= 0:
        return [0.0]
    return [(frequencies[0] * log(frequencies[0], 2)) +
            (frequencies[1] * log(frequencies[1], 2))]


def get_bit_flips(binary_string):
    """Count the four types of consecutive-bit transitions (normalised)."""
    total = len(binary_string)
    if total <= 1:
        return [0.0, 0.0, 0.0, 0.0]

    zz = zo = oz = oo = 0.0
    for i in range(total - 1):
        a, b = binary_string[i], binary_string[i + 1]
        if a == '0':
            if b == '0':
                zz += 1
            else:
                zo += 1
        else:
            if b == '0':
                oz += 1
            else:
                oo += 1
    return [zz / total, zo / total, oz / total, oo / total]


def get_self_correlation_sum(binary_string, max_shifts=50):
    """Calculate self-correlation sum (limited shifts for performance)."""
    t = len(binary_string)
    if t <= 1:
        return [0.0]

    ac = 0.0
    for i in range(1, min(t, max_shifts + 1)):
        shifted = binary_string[i:] + binary_string[:i]
        correct = sum(1 for a, b in zip(binary_string, shifted) if a == b)
        err = t - correct
        ac += float(correct - err) / t

    return [ac / min(t - 1, max_shifts)]


# ---------------------------------------------------------------------------
# Byte-level features
# ---------------------------------------------------------------------------

def get_byte_entropy(data_bytes):
    """Calculate byte-level Shannon entropy (0-8 bits)."""
    length = len(data_bytes)
    if length == 0:
        return 0.0
    freq = [0] * 256
    for b in data_bytes:
        freq[b] += 1
    prob = [f / length for f in freq]
    return float(scipy_entropy(prob, base=2))


def get_byte_distribution_stats(data_bytes):
    """Compute advanced byte-distribution statistics.

    Returns: [chi_squared, n_unique_bytes, quartile_25, median, quartile_75,
              iqr, skewness, kurtosis, max_byte_freq, min_nonzero_byte_freq]
    """
    length = len(data_bytes)
    if length == 0:
        return [0.0] * 10

    data_array = np.array(list(data_bytes), dtype=np.float64)

    # Chi-squared statistic against uniform distribution
    expected = length / 256.0
    freq = np.zeros(256)
    for b in data_bytes:
        freq[b] += 1
    chi_sq = float(np.sum((freq - expected) ** 2 / expected)) if expected > 0 else 0.0

    # Number of unique byte values (out of 256)
    n_unique = float(np.sum(freq > 0))

    # Quartiles of byte values
    q25 = float(np.percentile(data_array, 25))
    median = float(np.median(data_array))
    q75 = float(np.percentile(data_array, 75))
    iqr = q75 - q25

    # Higher-order moments
    if np.std(data_array) > 0:
        skewness = float(np.mean(((data_array - np.mean(data_array)) / np.std(data_array)) ** 3))
        kurtosis = float(np.mean(((data_array - np.mean(data_array)) / np.std(data_array)) ** 4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0

    # Max and min (non-zero) byte frequencies normalised
    max_freq = float(np.max(freq) / length)
    nonzero = freq[freq > 0]
    min_nz_freq = float(np.min(nonzero) / length) if len(nonzero) > 0 else 0.0

    return [chi_sq, n_unique, q25, median, q75, iqr, skewness, kurtosis,
            max_freq, min_nz_freq]


# ---------------------------------------------------------------------------
# Unified feature extraction
# ---------------------------------------------------------------------------

# Counts:  byte-basic(6) + byte-dist(10) + bin-freq(2) + bin-ent(1)
#         + bit-flips(4) + corr(1) + streaks(max_streaks+1)
NUM_BYTE_BASIC = 6
NUM_BYTE_DIST = 10
NUM_BIN_FIXED = 7           # freq(2) + entropy(1) + bit_flips(4)
NUM_CORR = 1

def feature_vector_size(max_streaks=10):
    """Return the expected length of a feature vector."""
    return (NUM_BYTE_BASIC + NUM_BYTE_DIST + NUM_BIN_FIXED +
            NUM_CORR + (max_streaks + 1))


def extract_features(data_bytes, max_streaks=10):
    """Extract all features from raw bytes (ciphertext or hash output).

    Feature layout (total = 6+10+2+1+4+1+(max_streaks+1) = 35 with max_streaks=10):
      Byte basic : length, byte_entropy, mean, std, length_mod_16, length_mod_8
      Byte dist  : chi_sq, n_unique, q25, median, q75, iqr, skew, kurt, max_freq, min_nz_freq
      Binary freq: freq_0, freq_1
      Binary ent : binary_entropy
      Bit flips  : zz, zo, oz, oo
      Correlation: self_correlation_sum
      Streaks    : streak_1 .. streak_max, total_streaks
    """
    length = len(data_bytes)

    if length == 0:
        return [0.0] * feature_vector_size(max_streaks)

    # Byte-level basic features
    data_array = np.array(list(data_bytes), dtype=np.uint8)
    byte_ent = get_byte_entropy(data_bytes)
    mean_val = float(np.mean(data_array))
    std_val = float(np.std(data_array))
    len_mod_16 = float(length % 16)
    len_mod_8 = float(length % 8)

    # Byte-level distribution features
    dist_stats = get_byte_distribution_stats(data_bytes)

    # Binary-string features
    binary_str = bytes_to_binary_string(data_bytes)
    frequencies = get_frequencies(binary_str)
    bin_entropy = get_entropy(frequencies)
    bit_flips = get_bit_flips(binary_str)
    self_corr = get_self_correlation_sum(binary_str)
    streaks = get_streaks(binary_str, max_streaks)

    features = ([float(length), byte_ent, mean_val, std_val, len_mod_16, len_mod_8] +
                dist_stats + frequencies + bin_entropy + bit_flips + self_corr + streaks)

    return features


def extract_features_from_hex(hex_string, max_streaks=10):
    """Convenience wrapper: extract features from a hex-encoded string."""
    try:
        data_bytes = bytes.fromhex(hex_string)
        return extract_features(data_bytes, max_streaks)
    except ValueError:
        return extract_features(b'', max_streaks)
