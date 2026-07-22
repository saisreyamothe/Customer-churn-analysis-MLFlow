"""
Drift detection using PSI (Population Stability Index).
Threshold: PSI > 0.2 triggers retraining alert.
"""

import pandas as pd
import numpy as np


def calculate_psi(expected_dist: np.ndarray, actual_dist: np.ndarray) -> float:
    """Calculate Population Stability Index."""
    def psi_bucket(expected, actual):
        if actual == 0:
            actual = 0.0001
        if expected == 0:
            expected = 0.0001
        return (actual - expected) * np.log(actual / expected)
    
    psi = sum(psi_bucket(e, a) for e, a in zip(expected_dist, actual_dist))
    return psi


def check_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.2) -> bool:
    """Check for data drift using PSI.
    
    PSI > 0.2: High drift - trigger retraining
    """
    drift_detected = False
    drift_features = []
    
    for col in baseline_df.select_dtypes(include=[np.number]).columns:
        # Bin data
        baseline_dist, _ = np.histogram(baseline_df[col], bins=10, density=True)
        current_dist, _ = np.histogram(current_df[col], bins=10, density=True)
        
        # Normalize
        baseline_dist = baseline_dist / baseline_dist.sum()
        current_dist = current_dist / current_dist.sum()
        
        # Calculate PSI
        psi = calculate_psi(baseline_dist, current_dist)
        
        if psi > threshold:
            drift_detected = True
            drift_features.append((col, psi))
    
    if drift_detected:
        print(f"⚠️  DRIFT DETECTED (PSI > {threshold})")
        for feature, psi_val in drift_features:
            print(f"   {feature}: PSI = {psi_val:.4f}")
        print("   → Trigger model retraining")
    
    return drift_detected


if __name__ == "__main__":
    # Example
    baseline = pd.DataFrame({
        "tenure": np.random.normal(30, 10, 1000),
        "charges": np.random.normal(65, 20, 1000),
    })
    
    # Distribution shift
    current = pd.DataFrame({
        "tenure": np.random.normal(25, 12, 1000),  # Shifted
        "charges": np.random.normal(70, 22, 1000),
    })
    
    check_drift(baseline, current)
