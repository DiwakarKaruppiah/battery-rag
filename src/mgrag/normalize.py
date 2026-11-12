"""Make numbers from different papers comparable.

Papers write the same thing many ways: 'mAh/g', 'mAh g−1', 'Ah kg-1'; '0.995' vs '99.5 %';
'0.2 V' vs '200 mV'; 'Mo₆S₈' vs 'Mo6S8'. Without this step, a property database
cannot rank or compare anything. Pure Python, unit-tested.
"""
from __future__ import annotations

import re

_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁻−–", "01234567890123456789---")

# property → canonical unit, and {normalized unit spelling: factor to canonical}
UNITS: dict[str, tuple[str, dict[str, float]]] = {
    "specific_capacity": ("mAh g-1", {"mah/g": 1, "mahg-1": 1, "ah/kg": 1, "ahkg-1": 1}),
    "areal_capacity": ("mAh cm-2", {"mah/cm2": 1, "mahcm-2": 1}),
    "coulombic_efficiency": ("%", {"%": 1, "percent": 1}),
    "overpotential": ("mV", {"mv": 1, "v": 1000}),
    "average_voltage": ("V", {"v": 1, "mv": 0.001, "vvs.mg/mg2+": 1, "vvsmg": 1}),
    "anodic_stability": ("V", {"v": 1, "vvs.mg/mg2+": 1, "vvsmg": 1, "vvs.mg": 1}),
    "ionic_conductivity": ("mS cm-1", {"ms/cm": 1, "mscm-1": 1, "s/cm": 1000, "scm-1": 1000}),
    "energy_density": ("Wh kg-1", {"wh/kg": 1, "whkg-1": 1}),
    "cycle_life": ("cycles", {"cycles": 1, "cycle": 1, "": 1}),
}


def unit_key(unit: str) -> str:
    """'mAh g⁻¹' → 'mahg-1';  'mS / cm' → 'ms/cm'."""
    u = (unit or "").translate(_SUB).lower()
    u = u.replace("·", "").replace("^", "").replace(" ", "").replace("per", "/")
    return u.replace("(", "").replace(")", "")


def normalize_value(prop: str, value: float, unit: str) -> tuple[float | None, str | None]:
    """Return (value in canonical unit, canonical unit), or (None, None) when unknown."""
    if prop not in UNITS:
        return None, None
    canon, table = UNITS[prop]
    key = unit_key(unit)
    if prop == "coulombic_efficiency" and key == "" and 0 < value <= 1:
        return value * 100, canon  # 0.995 written as a fraction
    for spelling, factor in table.items():
        if key == spelling or (spelling and key.startswith(spelling)):
            return value * factor, canon
    return None, None


def material_key(name: str) -> str:
    """Loose identity for grouping: 'Mo₆S₈ (Chevrel)' and 'mo6s8' → 'mo6s8'."""
    s = (name or "").translate(_SUB).lower()
    s = re.sub(r"\(.*?chevrel.*?\)|\bthe\b", "", s)
    s = re.sub(r"[\s_]+", "", s)
    return s.strip(" .,;:")
