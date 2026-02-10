#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Generator — generates synthetic encrypted / hashed data corpus.

Supports the following algorithms:
  Ciphers : AES-128, DES, Triple-DES, RC4, RC2, Blowfish, ChaCha20
  Hashes  : MD5, SHA-1, SHA-256, SHA-512

Each sample is a ciphertext (or hash digest) produced from a random plaintext.
Only the *ciphertext* features are kept, because at prediction time the user
only supplies ciphertext.
"""

import os
import sys
import numpy as np
import hashlib
from random import choice

from Crypto.Cipher import AES, DES, ARC4, ARC2, DES3, Blowfish, ChaCha20
from Crypto.Random import get_random_bytes

# Add script directory so sibling modules can be imported
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import FeatureExtraction as fe

# ── Algorithm ↔ label mapping ────────────────────────────────────────────────

ALGORITHM_LABELS = {
    'AES':       0,
    'DES':       1,
    'RC4':       2,
    'RC2':       3,
    'TripleDES': 4,
    'Blowfish':  5,
    'ChaCha20':  6,
    'MD5':       7,
    'SHA1':      8,
    'SHA256':    9,
    'SHA512':   10,
}

LABEL_NAMES = {v: k for k, v in ALGORITHM_LABELS.items()}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _pad(data, block_size):
    """PKCS7-style padding."""
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)


# ── Cipher encryption wrappers ───────────────────────────────────────────────

def encrypt_aes(plaintext):
    key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(_pad(plaintext, 16))


def encrypt_des(plaintext):
    key = get_random_bytes(8)
    cipher = DES.new(key, DES.MODE_ECB)
    return cipher.encrypt(_pad(plaintext, 8))


def encrypt_rc4(plaintext):
    key = get_random_bytes(16)
    cipher = ARC4.new(key)
    return cipher.encrypt(plaintext)


def encrypt_rc2(plaintext):
    key = get_random_bytes(16)
    cipher = ARC2.new(key, ARC2.MODE_ECB)
    return cipher.encrypt(_pad(plaintext, 8))


def encrypt_tripledes(plaintext):
    key = DES3.adjust_key_parity(get_random_bytes(24))
    cipher = DES3.new(key, DES3.MODE_ECB)
    return cipher.encrypt(_pad(plaintext, 8))


def encrypt_blowfish(plaintext):
    key = get_random_bytes(16)
    cipher = Blowfish.new(key, Blowfish.MODE_ECB)
    return cipher.encrypt(_pad(plaintext, 8))


def encrypt_chacha20(plaintext):
    key = get_random_bytes(32)
    cipher = ChaCha20.new(key=key)
    return cipher.nonce + cipher.encrypt(plaintext)


CIPHER_FUNCTIONS = {
    'AES':       encrypt_aes,
    'DES':       encrypt_des,
    'RC4':       encrypt_rc4,
    'RC2':       encrypt_rc2,
    'TripleDES': encrypt_tripledes,
    'Blowfish':  encrypt_blowfish,
    'ChaCha20':  encrypt_chacha20,
}

HASH_ALGORITHMS = {
    'MD5':    'md5',
    'SHA1':   'sha1',
    'SHA256': 'sha256',
    'SHA512': 'sha512',
}

# ── Sample generation ────────────────────────────────────────────────────────

def generate_cipher_samples(n_samples_per_algo, plaintext_sizes, max_streaks=10):
    """Return (features_list, labels_list) for all cipher algorithms."""
    features_list, labels_list = [], []

    for algo_name, encrypt_fn in CIPHER_FUNCTIONS.items():
        label = ALGORITHM_LABELS[algo_name]
        print(f"  Generating {n_samples_per_algo} samples for {algo_name}...")

        for _ in range(n_samples_per_algo):
            pt_size = choice(plaintext_sizes)
            plaintext = get_random_bytes(pt_size)
            try:
                ciphertext = encrypt_fn(plaintext)
                features = fe.extract_features(ciphertext, max_streaks)
                features_list.append(features)
                labels_list.append(label)
            except Exception as e:
                print(f"    Warning: {algo_name} encryption failed: {e}")

    return features_list, labels_list


def generate_hash_samples(n_samples_per_algo, plaintext_sizes, max_streaks=10):
    """Return (features_list, labels_list) for all hash algorithms."""
    features_list, labels_list = [], []

    for algo_name, hash_name in HASH_ALGORITHMS.items():
        label = ALGORITHM_LABELS[algo_name]
        print(f"  Generating {n_samples_per_algo} samples for {algo_name}...")

        for _ in range(n_samples_per_algo):
            pt_size = choice(plaintext_sizes)
            plaintext = get_random_bytes(pt_size)
            try:
                h = hashlib.new(hash_name)
                h.update(plaintext)
                digest = h.digest()
                features = fe.extract_features(digest, max_streaks)
                features_list.append(features)
                labels_list.append(label)
            except Exception as e:
                print(f"    Warning: {algo_name} hashing failed: {e}")

    return features_list, labels_list


# ── Legacy helpers (kept for compatibility with old callers) ─────────────────

def generate_corpus(synthetic_samples, synthetic_classes,
                    fname_train_samples, fname_train_classes,
                    fname_test_samples, fname_test_classes,
                    percent_train=0.8):
    """Split & save corpus to text files (numpy format)."""
    assert len(synthetic_samples) == len(synthetic_classes) and 0 <= percent_train <= 1
    p = np.random.permutation(len(synthetic_samples))
    synthetic_samples = synthetic_samples[p]
    synthetic_classes = synthetic_classes[p]
    split = int(percent_train * len(synthetic_samples))
    np.savetxt(fname_train_samples, synthetic_samples[:split])
    np.savetxt(fname_train_classes, synthetic_classes[:split])
    np.savetxt(fname_test_samples, synthetic_samples[split:])
    np.savetxt(fname_test_classes, synthetic_classes[split:])


def load_corpus(fname_train_samples, fname_train_classes,
                fname_test_samples, fname_test_classes):
    X_train = np.loadtxt(fname_train_samples)
    Y_train = np.loadtxt(fname_train_classes)
    X_test  = np.loadtxt(fname_test_samples)
    Y_test  = np.loadtxt(fname_test_classes)
    return X_train, Y_train, X_test, Y_test
