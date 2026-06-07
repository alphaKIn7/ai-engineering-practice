import httpx
import streamlit as st

API_URL = "http://127.0.0.1:8001/verify"

st.title("🔬 SciFact Claim Verifier")
claim = st.text_area("Enter a scientific claim:", height=80)

if st.button("Verify", type="primary") and claim.strip():
    try:
        with st.spinner("Verifying..."):
            resp = httpx.post(API_URL, json={"claim": claim}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        st.error("Could not reach the verifier service. Is uvicorn running?")
        st.stop()
    except httpx.HTTPStatusError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        st.stop()

    verdict = data["verdict"]

    # Verdict display
    icons = {"supported": "✅", "refuted": "❌", "not_enough_info": "❓"}
    icon = icons.get(verdict["verdict"], "")
    st.markdown(f"### {icon} {verdict['verdict'].replace('_', ' ').title()}")

    # Reasoning
    st.markdown(f"**Reasoning:** {verdict['reasoning']}")

    # Cited chunks
    if verdict["supporting_chunks"]:
        st.markdown(f"**Cited chunk IDs:** {verdict['supporting_chunks']}")

    # Retrieved evidence
    with st.expander("View retrieved evidence"):
        for i, chunk in enumerate(data["chunks"]):
            st.markdown(f"**Rank {i+1}** — score: {chunk['score']:.4f} — ID: {chunk['id']}")
            st.markdown(f"*{chunk['title']}*")
            st.caption(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
            st.divider()

    # Footer
    st.caption(f"Verified in {data['latency_ms'] / 1000:.1f}s")
