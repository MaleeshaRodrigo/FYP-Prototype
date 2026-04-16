"""Constant clinical thresholds — single source of truth for the API."""

CLINICAL_TARGETS = {
    "auc": 0.80,
    "balanced_accuracy": 0.65,
    "melanoma_sensitivity": 0.40,
    "non_mel_specificity": 0.82,
}

BORDERLINE_MARGIN = 0.05

PHASE_TRANSITION_THRESHOLD = 0.225

GA_DEFAULTS = {
    "alpha": 0.5467,
    "tau": 0.7671,
    "theta": 0.3985,
}
