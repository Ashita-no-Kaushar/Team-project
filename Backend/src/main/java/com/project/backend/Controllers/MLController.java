package com.project.backend.Controllers;


import com.project.backend.Entities.PredictionResult;
import com.project.backend.Services.MLService;
import com.project.backend.Services.UserService;
import com.project.backend.Services.GeneratedSampleRegistryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.LinkedHashMap;
import java.util.regex.Pattern;
import java.util.logging.Logger;

//
//@RestController
//@RequestMapping("/api/ml")
//public class MLController {
//    private final MLService mlService;
//
//
//    public MLController(MLService mlService) {
//        this.mlService = mlService;
//    }
//
//    private static final Logger log = Logger.getLogger(MLController.class.getName());
//
//    @PostMapping("/predict")
//    public String predict(@RequestBody Map<String, String> input) {
//        log.info("Received Request in the mlcontroller.");
//        String inputHex = input.get("input_hex");
//        Optional<String> result = mlService.predictAlgorithm(inputHex);
//        log.info("Predicted op from controller. : " + result.orElse(""));
//        return result.orElse("{\"error\": \"Prediction failed\"}");
//    }
//}


@RestController
@RequestMapping("/api/ml")
public class MLController {

    private final MLService mlService;

    private final UserService userService;

    private final GeneratedSampleRegistryService generatedSampleRegistryService;

    private static final Pattern HEX_PATTERN = Pattern.compile("^[0-9a-fA-F]+$");
    private static final Set<String> TRUSTED_ML_FALLBACK_LABELS = Set.of(
            "MD5", "SHA1", "SHA-1", "SHA256", "SHA-256", "SHA3-256", "SHA-3-256", "SHA512", "SHA-512"
    );
    private static final String REGISTRY_MODEL_VERSION = "registry-v1";


    @Autowired
    public MLController(MLService mlService,
                        UserService userService,
                        GeneratedSampleRegistryService generatedSampleRegistryService) {
        this.userService = userService;
        this.mlService = mlService;
        this.generatedSampleRegistryService = generatedSampleRegistryService;
    }



    private static final Logger log = Logger.getLogger(MLController.class.getName());

    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> predict(@RequestBody Map<String, String> input) {
        log.info("Received Request in the ML Controller.");

        if (!input.containsKey("input_hex")) {
            log.info("Input has no 'input_hex' key.");
            return ResponseEntity.badRequest().body(Map.of("error", "Missing 'input_hex' field"));
        }

        String inputHex = normalizeHex(input.get("input_hex"));
        if (!isValidHex(inputHex)) {
            return ResponseEntity.badRequest().body(Map.of("error", "Invalid hex input. Please provide even-length hexadecimal data."));
        }

        Optional<String> exactMatch = generatedSampleRegistryService.lookupAlgorithm(inputHex);
        if (exactMatch.isPresent()) {
            savePredictionSafely(inputHex, exactMatch.get());
            return ResponseEntity.ok(predictionResponse(
                exactMatch.get(),
                "exact_match",
                1.0,
                REGISTRY_MODEL_VERSION,
                null
            ));
        }

        Optional<PredictionResult> result = mlService.predictAlgorithm(inputHex);

        if (result.isPresent()) {
            log.info("result is present.");
            PredictionResult predictionResult = result.get();
            String predicted = predictionResult.predictedAlgorithm();
            if ("Unknown".equalsIgnoreCase(predicted)) {
                return ResponseEntity.ok(predictionResponse(
                    "Unknown",
                    predictionResult.source(),
                    predictionResult.confidence(),
                    predictionResult.modelVersion(),
                    "Low confidence prediction. Try a longer sample."
                ));
            }

            if (!isTrustedMlFallbackLabel(predicted)) {
                return ResponseEntity.ok(predictionResponse(
                    "Unknown",
                    predictionResult.source(),
                    predictionResult.confidence(),
                    predictionResult.modelVersion(),
                    "Model confidence is unreliable for this external cipher text. Generate sample via encryption endpoint for exact detection."
                ));
            }

            savePredictionSafely(inputHex, predicted);
            return ResponseEntity.ok(predictionResponse(
                predicted,
                predictionResult.source(),
                predictionResult.confidence(),
                predictionResult.modelVersion(),
                null
            ));
        } else {
            log.info("result is not present");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Prediction failed"));
        }
    }

    private void savePredictionSafely(String inputHex, String predicted) {
        log.info("Sending to user service from ml controller.");
        try {
            userService.savePrediction(inputHex, predicted);
        } catch (Exception e) {
            log.warning("Could not save prediction to user history: " + e.getMessage());
            // Don't fail the prediction just because saving failed
        }
    }

    private String normalizeHex(String value) {
        if (value == null) {
            return "";
        }
        return value.replaceAll("\\s+", "");
    }

    private boolean isValidHex(String value) {
        return !value.isBlank() && value.length() % 2 == 0 && HEX_PATTERN.matcher(value).matches();
    }

    private boolean isTrustedMlFallbackLabel(String predicted) {
        return predicted != null && TRUSTED_ML_FALLBACK_LABELS.contains(predicted.toUpperCase().replace('_', '-'));
    }

    private Map<String, Object> predictionResponse(String predictedAlgorithm,
                                                   String source,
                                                   Double confidence,
                                                   String modelVersion,
                                                   String warning) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("predicted_algorithm", predictedAlgorithm);

        if (source != null && !source.isBlank()) {
            payload.put("source", source);
        }
        if (confidence != null) {
            payload.put("confidence", confidence);
        }
        if (modelVersion != null && !modelVersion.isBlank()) {
            payload.put("model_version", modelVersion);
        }
        if (warning != null && !warning.isBlank()) {
            payload.put("warning", warning);
        }

        return payload;
    }
}

