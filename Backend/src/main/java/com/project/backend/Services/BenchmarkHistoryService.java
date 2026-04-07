package com.project.backend.Services;

import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

@Service
public class BenchmarkHistoryService {
    private static final int MAX_HISTORY_ENTRIES = 50;
    private static final int DEFAULT_LIMIT = 10;
    private static final int MAX_LIMIT = 50;
    private static final Set<String> SUPPORTED_MODES = Set.of("hash-only", "mixed", "strict");

    private final Deque<Map<String, Object>> entries = new ArrayDeque<>();

    public synchronized Map<String, Object> recordBenchmarkRun(Map<String, Object> report) {
        Map<String, Object> snapshot = new LinkedHashMap<>(report);
        snapshot.putIfAbsent("run_at", Instant.now().toString());

        Object modeValue = snapshot.get("mode");
        if (!(modeValue instanceof String mode) || mode.isBlank()) {
            snapshot.put("mode", "mixed");
        }

        entries.addFirst(snapshot);
        while (entries.size() > MAX_HISTORY_ENTRIES) {
            entries.removeLast();
        }

        return new LinkedHashMap<>(snapshot);
    }

    public synchronized List<Map<String, Object>> getHistory(String mode, Integer limit) {
        String normalizedMode = normalizeMode(mode);
        int boundedLimit = normalizeLimit(limit);

        List<Map<String, Object>> result = new ArrayList<>();
        for (Map<String, Object> entry : entries) {
            if (normalizedMode != null && !normalizedMode.equals(extractMode(entry))) {
                continue;
            }

            result.add(new LinkedHashMap<>(entry));
            if (result.size() >= boundedLimit) {
                break;
            }
        }

        return result;
    }

    private String extractMode(Map<String, Object> entry) {
        Object value = entry.get("mode");
        if (!(value instanceof String mode) || mode.isBlank()) {
            return "mixed";
        }
        String normalized = mode.trim().toLowerCase(Locale.ROOT);
        return SUPPORTED_MODES.contains(normalized) ? normalized : "mixed";
    }

    private String normalizeMode(String mode) {
        if (mode == null || mode.isBlank()) {
            return null;
        }

        String normalized = mode.trim().toLowerCase(Locale.ROOT);
        if ("all".equals(normalized)) {
            return null;
        }

        return SUPPORTED_MODES.contains(normalized) ? normalized : null;
    }

    private int normalizeLimit(Integer limit) {
        if (limit == null) {
            return DEFAULT_LIMIT;
        }
        if (limit < 1) {
            return 1;
        }
        return Math.min(limit, MAX_LIMIT);
    }
}
