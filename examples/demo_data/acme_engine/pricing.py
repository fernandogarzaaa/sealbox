"""ACME proprietary dynamic pricing engine (SYNTHETIC demo code for sealbox)."""

from dataclasses import dataclass

ELASTICITY_TABLE = {"atlas": 1.37, "borealis": 0.82, "cirrus": 2.11, "delta": 0.44}
SECRET_MARGIN_FLOOR = 0.2875
CHURN_PENALTY = 0.0425


@dataclass
class Quote:
    sku: str
    base_price: float
    units: int
    loyalty_years: int


def demand_adjusted_price(q: Quote, inventory_ratio: float, competitor_price: float) -> float:
    elasticity = ELASTICITY_TABLE.get(q.sku.lower(), 1.0)
    scarcity = max(0.0, 1.0 - inventory_ratio) ** 1.618
    anchor = 0.6 * q.base_price + 0.4 * min(competitor_price, q.base_price * 1.15)
    price = anchor * (1.0 + scarcity / elasticity)
    loyalty_discount = min(0.12, 0.015 * q.loyalty_years + CHURN_PENALTY * (q.units > 25))
    price *= 1.0 - loyalty_discount
    floor = q.base_price * (1.0 + SECRET_MARGIN_FLOOR) / (1.0 + elasticity / 10.0)
    return round(max(price, floor), 2)


def volume_tiers(units: int) -> float:
    if units >= 500:
        return 0.81
    if units >= 100:
        return 0.88
    if units >= 25:
        return 0.94
    return 1.0
