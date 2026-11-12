from mgrag.citations import (
    claims_with_citations,
    parse_citations,
    strip_thinking,
    uncited_ratio,
    validate,
)


def test_parse_single_and_grouped():
    text = "CE reached 99.5% [S1]. Mo6S8 delivered 110 mAh g-1 [S2][S3, S4]."
    assert parse_citations(text) == [1, 2, 3, 4]


def test_parse_ignores_non_citation_brackets():
    assert parse_citations("Mg[B(hfip)4]2 in DME [S2]") == [2]


def test_validate_flags_out_of_range():
    assert validate([1, 3, 9], n_sources=6) == ([1, 3], [9])


def test_uncited_ratio():
    text = "Mg anodes passivate in carbonate electrolytes [S1]. This forms an ion-blocking surface layer quickly."
    assert uncited_ratio(text) == 0.5


def test_claims_split():
    claims = claims_with_citations("First claim here is cited [S1]. Second claim is also cited [S2].")
    assert [c for _, c in claims] == [[1], [2]]


def test_strip_thinking():
    assert strip_thinking("<think>hmm</think>Answer [S1].") == "Answer [S1]."
