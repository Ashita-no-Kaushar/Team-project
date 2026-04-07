package com.project.backend.Services;

import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

@Service
public class GeneratedSampleRegistryService {

    private static final int MAX_ENTRIES = 5000;

    private final Map<String, String> sampleToAlgorithm = Collections.synchronizedMap(
            new LinkedHashMap<>(128, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<String, String> eldest) {
                    return size() > MAX_ENTRIES;
                }
            }
    );

    public void registerSample(String hexPayload, String algorithm) {
        String normalizedHex = normalizeHex(hexPayload);
        if (normalizedHex == null || algorithm == null || algorithm.isBlank()) {
            return;
        }
        sampleToAlgorithm.put(normalizedHex, algorithm);
    }

    public Optional<String> lookupAlgorithm(String hexPayload) {
        String normalizedHex = normalizeHex(hexPayload);
        if (normalizedHex == null) {
            return Optional.empty();
        }
        return Optional.ofNullable(sampleToAlgorithm.get(normalizedHex));
    }

    private String normalizeHex(String value) {
        if (value == null) {
            return null;
        }

        String normalized = value.replaceAll("\\s+", "").toLowerCase();
        if (normalized.isEmpty() || normalized.length() % 2 != 0) {
            return null;
        }

        for (int i = 0; i < normalized.length(); i++) {
            char c = normalized.charAt(i);
            boolean isHex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            if (!isHex) {
                return null;
            }
        }

        return normalized;
    }
}
