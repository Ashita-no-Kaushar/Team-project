package com.project.backend.Entities;

public record PredictionResult(
        String predictedAlgorithm,
        Double confidence,
        String modelVersion,
        String source
) {
}
