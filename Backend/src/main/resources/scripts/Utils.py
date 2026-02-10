#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for cryptographic data generation."""

import os


def generate_random_string(length):
    """Generate a random byte string of the given length.

    Args:
        length: Number of random bytes to generate.

    Returns:
        A bytes object of the specified length.
    """
    return os.urandom(length)
