"""Run:  streamlit run app/streamlit_app.py

Three modes:
  Ask       cited answers from text, tables and figures (RAG + guardrails)
  Compare   the literature as a database: rank values, see where papers disagree, and why
  Discover  coverage map of what your library measures, and suggested research gaps
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from mgrag.config import settings
from mgrag.evidence import coverage_matrix, explain_spread, find_spreads, load_properties, suggest_gaps
from mgrag.rag import answer, source_label
from mgrag.sqltool import ask_sql, route

st.set_page_config(page_title="MgScholar — Mg battery research copilot", page_icon="🔋", layout="wide")
st.title("🔋 MgScholar")
st.caption("A research copilot for your magnesium-battery library: cited answers, comparable numbers, research gaps.")

with st.sidebar:
    st.subheader("Settings")
    mode = st.radio("Retrieval", ["hybrid", "dense", "sparse"], index=0)
    use_rerank = st.toggle("Cross-encoder reranker", value=True)
    threshold = st.slider("Abstain below rerank score", 0.0, 1.0, settings.min_rerank_score, 0.05)
    top_k = st.slider("Sources given to the LLM", 3, 12, settings.top_k_context)
    st.caption(f"LLM: `{settings.llm_model}`  \nReranker: `{settings.reranker_model}`")

props = load_properties()
tab_ask, tab_compare, tab_discover = st.tabs(["Ask", "Compare", "Discover"])


def show_sql(question: str) -> None:
    try:
        sql, res = ask_sql(question, props)
    except Exception as e:  # show the reason, never crash the app
        st.info(f"Database view unavailable: {e}")
        return
    st.dataframe(res, use_container_width=True, hide_index=True)
    with st.expander("SQL used"):
        st.code(sql, language="sql")


# ---------------------------------------------------------------- Ask
with tab_ask:
    examples = [
        "Which electrolytes enable reversible Mg plating with Coulombic efficiency above 99%?",
        "Why do Chevrel-phase Mo6S8 cathodes show good Mg2+ kinetics?",
        "What causes passivation of Mg metal anodes in conventional carbonate electrolytes?",
    ]
    q = st.text_input("Question", placeholder=examples[0])
    st.caption("Try: " + " · ".join(examples[1:]))
    if q:
        with st.spinner("Retrieving, reranking, generating…"):
            r = answer(q, mode=mode, use_rerank=use_rerank, min_score=threshold, top_k=top_k)
        if r.abstained:
            st.warning(r.answer)
            st.caption(f"Guardrail: {r.reason}")
        else:
            st.markdown(re.sub(r"\[(S\d+)\]", r"**[\1]**", r.answer))
            if r.invalid_citations:
                st.error(f"Model cited non-existent sources: {r.invalid_citations}")
            if r.uncited_ratio > 0:
                st.caption(f"⚠️ {r.uncited_ratio:.0%} of sentences have no citation")
        cols = st.columns(len(r.timings) or 1)
        for col, (k, v) in zip(cols, r.timings.items(), strict=False):
            col.metric(k.replace("_s", ""), f"{v:.2f}s")

        if route(q) == "sql" and not props.empty:
            st.subheader("From the property database")
            st.caption("This looks like a numeric question, so here are the matching measurements too.")
            show_sql(q)

        st.subheader("Sources")
        for i, s in enumerate(r.sources, 1):
            mark = "✅ cited" if i in r.cited else ""
            score = s.get("rerank_score", s.get("retrieval_score", 0))
            with st.expander(f"[S{i}] {source_label(s)} · score {score:.2f} {mark}"):
                if s["type"] == "figure" and s.get("image_path") and Path(s["image_path"]).exists():
                    st.image(s["image_path"], use_container_width=True)
                    if s.get("description"):
                        st.caption(s["description"])
                st.markdown(s["text"])

# ---------------------------------------------------------------- Compare
with tab_compare:
    if props.empty:
        st.info("Run `python -m mgrag.cli extract` to build the property database.")
    else:
        st.markdown("**Ask the database** — rankings, filters, averages. Every row keeps its paper and page.")
        sq = st.text_input("Numeric question",
                           placeholder="Top 5 cathodes by specific capacity, with the current density used")
        if sq:
            show_sql(sq)

        st.markdown("**Explore**")
        c1, c2 = st.columns(2)
        sel_p = c1.multiselect("Property", sorted(props["property"].dropna().unique()))
        sel_c = c2.multiselect("Component", sorted(props["component"].dropna().unique()))
        view = props
        if sel_p:
            view = view[view["property"].isin(sel_p)]
        if sel_c:
            view = view[view["component"].isin(sel_c)]
        st.dataframe(view, use_container_width=True, hide_index=True)
        if len(sel_p) == 1 and view["value_norm"].notna().any():
            import plotly.express as px
            v = view.dropna(subset=["value_norm"])
            fig = px.strip(v, x="year", y="value_norm", color="component",
                           hover_data=["material", "conditions", "paper_id", "page"],
                           title=f"{sel_p[0].replace('_', ' ')} across your library ({v['unit_norm'].iloc[0]})")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Where papers disagree** — same material and property, notably different values")
        spreads = find_spreads(props)
        if not spreads:
            st.caption("No notable spreads yet (needs ≥ 2 papers reporting the same material + property).")
        for i, sp in enumerate(spreads[:15]):
            title = (f"{sp['material']} · {sp['component']} · {sp['property'].replace('_', ' ')}: "
                     f"{sp['min']:.3g}–{sp['max']:.3g} {sp['unit']} across {sp['n_papers']} papers")
            with st.expander(title):
                st.dataframe(pd.DataFrame(sp["rows"])[["value", "unit", "conditions", "paper_id", "page", "year"]],
                             use_container_width=True, hide_index=True)
                if st.button("Explain the difference", key=f"explain{i}"):
                    with st.spinner("Reading both sources…"):
                        st.markdown(explain_spread(sp))

# ---------------------------------------------------------------- Discover
with tab_discover:
    if props.empty:
        st.info("Run `python -m mgrag.cli extract` first.")
    else:
        m = coverage_matrix(props)
        st.markdown("**Coverage map** — how many of your papers report each property for each component. "
                    "Pale cells are candidate gaps (or gaps in your reading list).")
        import plotly.express as px
        fig = px.imshow(m, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        labels={"color": "papers"})
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Suggest research gaps"):
            with st.spinner("Reading the map…"):
                st.markdown(suggest_gaps(props))
            st.caption("Hypotheses for you to judge, grounded in your library — not established facts.")
