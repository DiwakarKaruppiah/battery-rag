import pandas as pd

from mgrag.evidence import coverage_matrix, find_spreads
from mgrag.normalize import material_key, normalize_value, unit_key
from mgrag.sqltool import is_safe, route

# ---------- unit + material normalization


def test_unit_key_handles_unicode_and_spacing():
    assert unit_key("mAh g⁻¹") == "mahg-1"
    assert unit_key("mS / cm") == "ms/cm"


def test_capacity_spellings_agree():
    assert normalize_value("specific_capacity", 120, "mAh/g") == (120, "mAh g-1")
    assert normalize_value("specific_capacity", 120, "mAh g−1") == (120, "mAh g-1")
    assert normalize_value("specific_capacity", 120, "Ah kg-1") == (120, "mAh g-1")


def test_conversions():
    assert normalize_value("overpotential", 0.2, "V") == (200, "mV")
    assert normalize_value("ionic_conductivity", 0.0056, "S/cm") == (5.6, "mS cm-1")
    assert normalize_value("coulombic_efficiency", 0.995, "") == (99.5, "%")


def test_unknown_unit_is_not_guessed():
    assert normalize_value("specific_capacity", 3, "mAh cm-2") == (None, None)
    assert normalize_value("other", 1, "x") == (None, None)


def test_material_key():
    assert material_key("Mo₆S₈ (Chevrel phase)") == material_key("mo6s8") == "mo6s8"


# ---------- evidence spreads + coverage


def _df():
    rows = [
        # CE for the same electrolyte: 97.0 vs 99.6 → 2.6 pct-points apart → notable
        ("Mg(TFSI)2-MgCl2/DME", "electrolyte", "coulombic_efficiency", 97.0, "%", "0.5 mA cm-2", "A2020", 4),
        ("mg(tfsi)2-mgcl2/dme", "electrolyte", "coulombic_efficiency", 99.6, "%", "1 mA cm-2", "B2022", 6),
        # capacity for Mo6S8: 100 vs 110 → 10 % apart → not notable
        ("Mo6S8", "cathode", "specific_capacity", 100, "mAh g-1", "0.1C", "A2020", 5),
        ("Mo₆S₈", "cathode", "specific_capacity", 110, "mAh g-1", "0.1C", "C2023", 3),
        # single paper → never a spread
        ("MnO2", "cathode", "specific_capacity", 250, "mAh g-1", "", "D2024", 2),
    ]
    df = pd.DataFrame(rows, columns=["material", "component", "property", "value", "unit",
                                     "conditions", "paper_id", "page"])
    df["material_key"] = df["material"].map(material_key)
    df["value_norm"], df["unit_norm"] = df["value"], df["unit"]
    df["chunk_id"] = df["paper_id"] + "::t1"
    return df


def test_find_spreads_flags_only_notable_multi_paper_groups():
    spreads = find_spreads(_df())
    assert len(spreads) == 1
    s = spreads[0]
    assert s["property"] == "coulombic_efficiency"
    assert (s["min"], s["max"], s["n_papers"]) == (97.0, 99.6, 2)


def test_coverage_counts_distinct_papers():
    m = coverage_matrix(_df())
    assert m.loc["cathode", "specific_capacity"] == 3
    assert m.loc["electrolyte", "specific_capacity"] == 0


# ---------- SQL safety + routing


def test_sql_guard():
    assert is_safe("SELECT material, value_norm FROM measurements ORDER BY value_norm DESC LIMIT 5")
    assert is_safe("with t as (select * from measurements) select * from t;")
    assert not is_safe("DROP TABLE measurements")
    assert not is_safe("SELECT 1; DELETE FROM measurements")
    assert not is_safe("SELECT * FROM read_csv_auto('/etc/passwd')")


def test_router():
    assert route("Top 5 cathodes by specific capacity") == "sql"
    assert route("Which electrolytes show Coulombic efficiency above 99%?") == "sql"
    assert route("Why do Mg anodes passivate in carbonate electrolytes?") == "rag"
