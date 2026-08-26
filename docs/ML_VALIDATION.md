# ML Validation

ML components must be evaluated on reproducible, versioned datasets representative of the target deployment.

Required metrics include precision, recall, F1, per-class performance, false-positive rate, false-negative rate and a confusion matrix. Record dataset version, split strategy, model/version, preprocessing, thresholds, hardware/runtime and evaluation code.

The repository must not claim a numerical accuracy or F1 value without reproducible evidence. Demo/synthetic data is not evidence of production performance.

Human review remains required for consequential privacy decisions. Model changes require regression evaluation and approval before production rollout.
