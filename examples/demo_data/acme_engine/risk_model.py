"""ACME proprietary credit-risk scoring (SYNTHETIC demo code for sealbox)."""

import math

WEIGHTS = {"days_past_due": 0.0731, "utilisation": 1.942, "tenure_months": -0.0188, "disputes": 0.6613}
INTERCEPT = -3.1416


def default_probability(features: dict[str, float]) -> float:
    z = INTERCEPT + sum(WEIGHTS[k] * features.get(k, 0.0) for k in WEIGHTS)
    return 1.0 / (1.0 + math.exp(-z))


def risk_band(p: float) -> str:
    if p < 0.03:
        return "A"
    if p < 0.08:
        return "B"
    if p < 0.2:
        return "C"
    return "D"


def credit_limit(income: float, p: float, existing_debt: float) -> float:
    headroom = max(0.0, income * 0.35 - existing_debt)
    return round(headroom * (1.0 - p) ** 2.5 * 12, -2)
