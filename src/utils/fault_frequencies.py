"""Fault frequency calculation utilities for bearing diagnosis."""
import math
from typing import Dict


def compute_fault_frequencies(
    N_balls: int,
    d_mm: float,
    D_mm: float,
    alpha_deg: float,
    f_shaft_hz: float
) -> Dict[str, float]:
    """
    Compute canonical bearing fault frequencies.

    Parameters:
        N_balls: Number of rolling elements
        d_mm: Rolling element diameter (mm)
        D_mm: Pitch diameter (mm)
        alpha_deg: Contact angle (degrees)
        f_shaft_hz: Shaft rotation frequency (Hz)

    Returns:
        Dictionary with fault frequencies in Hz:
        {BPFO, BPFI, BSF, FTF, f_s, 2f_s, 3f_s}
    """
    alpha_rad = math.radians(alpha_deg)
    d_D_ratio = d_mm / D_mm
    cos_alpha = math.cos(alpha_rad)

    # Core fault frequencies
    BPFO = (N_balls / 2) * f_shaft_hz * (1 - d_D_ratio * cos_alpha)
    BPFI = (N_balls / 2) * f_shaft_hz * (1 + d_D_ratio * cos_alpha)
    BSF = (D_mm / (2 * d_mm)) * f_shaft_hz * (1 - (d_D_ratio * cos_alpha) ** 2)
    FTF = (f_shaft_hz / 2) * (1 - d_D_ratio * cos_alpha)

    return {
        "BPFO": float(BPFO),
        "BPFI": float(BPFI),
        "BSF": float(BSF),
        "FTF": float(FTF),
        "f_s": float(f_shaft_hz),
        "2f_s": float(2 * f_shaft_hz),
        "3f_s": float(3 * f_shaft_hz),
    }


# Pre-computed and verified test cases for unit tests
CWRU_6205_TEST_CASE = {
    "N_balls": 9,
    "d_mm": 7.94,
    "D_mm": 39.04,
    "alpha_deg": 0.0,
    "f_shaft_hz": 29.95,
    "expected": {
        "BPFO": 107.37,
        "BPFI": 162.13,
        "BSF": 70.59,
        "FTF": 11.92,
    }
}

PU_6206_TEST_CASE = {
    "N_balls": 9,
    "d_mm": 7.938,
    "D_mm": 38.5,
    "alpha_deg": 0.0,
    "f_shaft_hz": 15.0,
    "expected": {
        "BPFO": 53.57,
        "BPFI": 81.42,
        "BSF": 34.83,
        "FTF": 5.96,
    }
}

JNU_ER16K_TEST_CASE = {
    "N_balls": 8,
    "d_mm": 7.5,
    "D_mm": 38.5,
    "alpha_deg": 0.0,
    "f_shaft_hz": 16.67,
    "expected": {
        "BPFO": 53.68,
        "BPFI": 79.64,
        "BSF": 41.15,
        "FTF": 6.72,
    }
}
