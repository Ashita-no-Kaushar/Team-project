package com.project.backend.Services;



import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.project.backend.Entities.PredictionResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.TimeUnit;

//@Service
//public class MLService {
//
//    private static final Logger log = LoggerFactory.getLogger(MLService.class);
//    private static final String PYTHON_SCRIPT = "src/main/resources/scripts/predict.py";
//
//    public Optional<String> predictAlgorithm(String inputHex) {
//        log.info("Received request to predict algorithm in the ML SERVICE.");
//        try {
//            log.info("Inside the try block of predictAlgorithm");
//            // Construct command: python3 predict.py "HEX_DATA"
//            ProcessBuilder processBuilder = new ProcessBuilder("python3", PYTHON_SCRIPT, inputHex);
//            processBuilder.redirectErrorStream(true);
//            Process process = processBuilder.start();
//            log.info("Reaced midway of process.");
//            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
//            StringBuilder output = new StringBuilder();
//            String line;
//            while ((line = reader.readLine()) != null) {
//                output.append(line);
//            }
//            log.info("Finally going to return the predicted algorithm.");
//
//            return Optional.of(output.toString());  // Return JSON response
//        } catch (Exception e) {
//            log.info("Exception caught while trying to predict algo in its service.");
//            e.printStackTrace();
//            return Optional.empty();
//        }
//    }
//}
//
@Service
public class MLService {
    private static final Logger log = LoggerFactory.getLogger(MLService.class);
    private static final String PYTHON_SCRIPT = "src/main/resources/scripts/predict.py";
    private static final long PYTHON_TIMEOUT_SECONDS = 30;
    private static final String DEFAULT_MODEL_VERSION = "legacy-unknown";
    private static final String DEFAULT_RUNTIME_MODEL_VERSION = "hybrid-rf-lr-v2";
    private static final double DEFAULT_CONFIDENCE_THRESHOLD = 0.75;
    private static final String MODEL_ARTIFACT = "model.pickle";
    private static final String LABEL_MAP_ARTIFACT = "label_map.pickle";
    private static final String MODE_HASH_ONLY = "hash-only";
    private static final String MODE_MIXED = "mixed";
    private static final String MODE_STRICT = "strict";
    private static final Set<String> SUPPORTED_BENCHMARK_MODES = Set.of(MODE_HASH_ONLY, MODE_MIXED, MODE_STRICT);
    private static final List<String> BENCHMARK_MODES = List.of(MODE_HASH_ONLY, MODE_MIXED, MODE_STRICT);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    public Optional<PredictionResult> predictAlgorithm(String inputHex) {
        log.info("Received request to predict algorithm in the ML SERVICE.");
        try {
            log.info("Inside the try block of predictAlgorithm");

            // Detect OS and use the correct Python command
            String pythonCommand = System.getProperty("os.name").toLowerCase().contains("win") ? "python" : "python3";
            String scriptPath = Paths.get(PYTHON_SCRIPT).toAbsolutePath().normalize().toString();

            ProcessBuilder processBuilder = new ProcessBuilder(pythonCommand, scriptPath, inputHex);
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();

            String output = readOutput(process.getInputStream());

            boolean completed = process.waitFor(PYTHON_TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!completed) {
                process.destroyForcibly();
                log.error("Python prediction timed out after {} seconds", PYTHON_TIMEOUT_SECONDS);
                return Optional.empty();
            }

            int exitCode = process.exitValue();
            if (exitCode != 0) {
                log.error("Python script exited with error code: {}. Output: {}", exitCode, output);
                return Optional.empty();
            }

            String result = output.lines()
                    .map(String::trim)
                    .filter(line -> !line.isEmpty())
                    .reduce((first, second) -> second)
                    .orElse("");

            if (result.isBlank() || result.startsWith("Error:")) {
                log.error("Python script returned invalid prediction output: {}", output);
                return Optional.empty();
            }

            Optional<PredictionResult> predictionResult = parsePredictionOutput(result);
            predictionResult.ifPresent(value ->
                    log.info("Returning predicted algorithm: {}, source: {}, modelVersion: {}, confidence: {}",
                            value.predictedAlgorithm(), value.source(), value.modelVersion(), value.confidence())
            );
            return predictionResult;
        } catch (Exception e) {
            log.error("Exception caught while predicting algorithm.", e);
            return Optional.empty();
        }
    }

    public Map<String, Object> getModelInfo() {
        Path scriptPath = Paths.get(PYTHON_SCRIPT).toAbsolutePath().normalize();
        Path scriptDirectory = scriptPath.getParent();
        Path modelPath = scriptDirectory.resolve(MODEL_ARTIFACT);
        Path labelMapPath = scriptDirectory.resolve(LABEL_MAP_ARTIFACT);

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("model_version", resolveRuntimeModelVersion());
        payload.put("confidence_threshold", resolveConfidenceThreshold());
        payload.put("python_script", scriptPath.toString());
        payload.put("model_artifact_exists", Files.exists(modelPath));
        payload.put("label_map_exists", Files.exists(labelMapPath));
        payload.put("timeout_seconds", PYTHON_TIMEOUT_SECONDS);
        payload.put("exact_match_source", "exact_match");
        payload.put("fallback_source", "ml_fallback");
        return payload;
    }

    public Map<String, Object> runQuickBenchmark() {
        return runQuickBenchmark(MODE_MIXED);
    }

    public Map<String, Object> runQuickBenchmark(String requestedMode) {
        String mode = resolveBenchmarkMode(requestedMode);
        long startedAt = System.nanoTime();

        List<BenchmarkInput> benchmarkInputs = buildBenchmarkInputsForMode(mode);

        List<Map<String, Object>> caseResults = new ArrayList<>();
        int passedCases = 0;
        double confidenceSum = 0.0;
        int confidenceCount = 0;

        for (BenchmarkInput benchmarkInput : benchmarkInputs) {
            Map<String, Object> casePayload = new LinkedHashMap<>();
            casePayload.put("name", benchmarkInput.name());
            casePayload.put("expected_algorithm", benchmarkInput.expectedAlgorithm());

            Optional<PredictionResult> predictionResult = predictAlgorithm(benchmarkInput.inputHex());
            if (predictionResult.isEmpty()) {
                casePayload.put("predicted_algorithm", "PredictionFailed");
                casePayload.put("source", "service_error");
                casePayload.put("match", false);
                caseResults.add(casePayload);
                continue;
            }

            PredictionResult prediction = predictionResult.get();
            casePayload.put("predicted_algorithm", prediction.predictedAlgorithm());
            casePayload.put("source", prediction.source());
            casePayload.put("model_version", prediction.modelVersion());

            if (prediction.confidence() != null) {
                casePayload.put("confidence", prediction.confidence());
                confidenceSum += prediction.confidence();
                confidenceCount++;
            }

            boolean match = labelsMatch(prediction.predictedAlgorithm(), benchmarkInput.expectedAlgorithm());
            casePayload.put("match", match);
            if (match) {
                passedCases++;
            }

            caseResults.add(casePayload);
        }

        long durationMs = (System.nanoTime() - startedAt) / 1_000_000;

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("mode", mode);
        report.put("available_modes", BENCHMARK_MODES);
        report.put("run_at", Instant.now().toString());
        report.put("model_info", getModelInfo());
        report.put("total_cases", benchmarkInputs.size());
        report.put("passed_cases", passedCases);
        report.put("pass_rate", benchmarkInputs.isEmpty() ? 0.0 : (double) passedCases / benchmarkInputs.size());
        report.put("duration_ms", durationMs);
        if (confidenceCount > 0) {
            report.put("avg_confidence", confidenceSum / confidenceCount);
        }
        report.put("results", caseResults);
        return report;
    }

    private List<BenchmarkInput> buildBenchmarkInputsForMode(String mode) {
        List<BenchmarkInput> hashOnlyInputs = List.of(
                new BenchmarkInput("MD5(hello)", "5d41402abc4b2a76b9719d911017c592", "MD5"),
                new BenchmarkInput("SHA1(hello)", "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d", "SHA-1"),
                new BenchmarkInput("SHA256(hello)", "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", "SHA-256"),
                new BenchmarkInput("SHA512(hello)", "9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca72323c3d99ba5c11d7c7acc6e14b8c5da0c4663475c2e5c3adef46f73bcdec043", "SHA-512")
        );

        if (MODE_HASH_ONLY.equals(mode)) {
            return hashOnlyInputs;
        }

        List<BenchmarkInput> mixedInputs = new ArrayList<>(hashOnlyInputs);
        mixedInputs.add(new BenchmarkInput("RandomHex-128", "00112233445566778899aabbccddeeff", "Unknown"));

        if (MODE_STRICT.equals(mode)) {
            mixedInputs.add(new BenchmarkInput("RandomHex-LongA", "4f3c2a1b9e8d7c6b5a493827160f1e2d3c4b5a69788796a5b4c3d2e1f0a9b8c7", "Unknown"));
            mixedInputs.add(new BenchmarkInput("RandomHex-LongB", "9f8e7d6c5b4a39281726354433221100ffeeddccbbaa99887766554433221100", "Unknown"));
        }

        return mixedInputs;
    }

    private String resolveBenchmarkMode(String requestedMode) {
        if (requestedMode == null || requestedMode.isBlank()) {
            return MODE_MIXED;
        }

        String normalized = requestedMode.trim().toLowerCase(Locale.ROOT);
        return SUPPORTED_BENCHMARK_MODES.contains(normalized) ? normalized : MODE_MIXED;
    }

    private String readOutput(InputStream inputStream) throws IOException {
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (output.length() > 0) {
                    output.append('\n');
                }
                output.append(line);
            }
        }
        return output.toString();
    }

    private Optional<PredictionResult> parsePredictionOutput(String rawResult) {
        try {
            if (rawResult.startsWith("{")) {
                JsonNode node = OBJECT_MAPPER.readTree(rawResult);
                String predictedAlgorithm = node.path("predicted_algorithm").asText("").trim();
                if (predictedAlgorithm.isBlank()) {
                    return Optional.empty();
                }

                Double confidence = null;
                JsonNode confidenceNode = node.get("confidence");
                if (confidenceNode != null && !confidenceNode.isNull()) {
                    confidence = confidenceNode.asDouble();
                }

                String modelVersion = node.path("model_version").asText(DEFAULT_MODEL_VERSION);
                String source = node.path("source").asText("ml_fallback");

                return Optional.of(new PredictionResult(predictedAlgorithm, confidence, modelVersion, source));
            }

            return Optional.of(new PredictionResult(rawResult, null, DEFAULT_MODEL_VERSION, "ml_fallback"));
        } catch (Exception parseError) {
            log.warn("Could not parse structured prediction output. Falling back to raw label: {}", rawResult);
            if (rawResult.isBlank()) {
                return Optional.empty();
            }
            return Optional.of(new PredictionResult(rawResult, null, DEFAULT_MODEL_VERSION, "ml_fallback"));
        }
    }

    private String resolveRuntimeModelVersion() {
        String configured = System.getenv("PREDICTION_MODEL_VERSION");
        if (configured == null || configured.isBlank()) {
            return DEFAULT_RUNTIME_MODEL_VERSION;
        }
        return configured.trim();
    }

    private double resolveConfidenceThreshold() {
        String configured = System.getenv("PREDICTION_CONFIDENCE_THRESHOLD");
        if (configured == null || configured.isBlank()) {
            return DEFAULT_CONFIDENCE_THRESHOLD;
        }

        try {
            return Double.parseDouble(configured.trim());
        } catch (NumberFormatException ex) {
            log.warn("Invalid PREDICTION_CONFIDENCE_THRESHOLD='{}'. Falling back to {}", configured, DEFAULT_CONFIDENCE_THRESHOLD);
            return DEFAULT_CONFIDENCE_THRESHOLD;
        }
    }

    private boolean labelsMatch(String predicted, String expected) {
        return normalizeLabel(predicted).equals(normalizeLabel(expected));
    }

    private String normalizeLabel(String label) {
        if (label == null || label.isBlank()) {
            return "";
        }

        String normalized = label.trim().toUpperCase(Locale.ROOT).replace('_', '-');
        return switch (normalized) {
            case "SHA1" -> "SHA-1";
            case "SHA256" -> "SHA-256";
            case "SHA512" -> "SHA-512";
            case "SHA3-256" -> "SHA-3-256";
            case "TRIPLEDES" -> "3DES";
            default -> normalized;
        };
    }

    private record BenchmarkInput(String name, String inputHex, String expectedAlgorithm) {
    }
}
