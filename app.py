import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import json
import time
from datetime import timedelta

st.set_page_config(page_title="Money Muling Detector", layout="wide")
st.title("💸 Money Muling Detection System")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:

    start_time = time.time()
    df = pd.read_csv(uploaded_file)

    required = {"transaction_id","sender_id","receiver_id","amount","timestamp"}
    if not required.issubset(df.columns):
        st.error("CSV must contain: transaction_id, sender_id, receiver_id, amount, timestamp")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    st.subheader("📄 Transaction Data")
    st.dataframe(df)

    G = nx.DiGraph()

    for _, r in df.iterrows():
        G.add_edge(r["sender_id"], r["receiver_id"], amount=r["amount"], timestamp=r["timestamp"])

    fraud_rings = []
    suspicious_accounts = set()
    ring_counter = 1

    risk_map = {
        "circular_flow": 90,
        "smurfing_fan_in": 85,
        "smurfing_fan_out": 85,
        "layered_shell_laundering": 95
    }

    def create_ring(members, pattern):
        nonlocal_counter[0] += 1

    # workaround counter (clean python)
    nonlocal_counter = [0]

    def add_ring(members, pattern):
        nonlocal_counter[0] += 1
        ring_id = f"RING_{nonlocal_counter[0]:03d}"

        fraud_rings.append({
            "ring_id": ring_id,
            "member_accounts": list(members),
            "pattern_type": pattern,
            "risk_score": float(risk_map[pattern])
        })

        for m in members:
            suspicious_accounts.add(m)

    # =========================
    # 1️⃣ CIRCULAR FLOWS
    # =========================
    for cycle in nx.simple_cycles(G):
        if 3 <= len(cycle) <= 5:
            add_ring(cycle, "circular_flow")

    # =========================
    # 2️⃣ SMURFING
    # =========================
    window = timedelta(hours=72)

    for acc in G.nodes():

        incoming = df[df["receiver_id"] == acc]
        outgoing = df[df["sender_id"] == acc]

        if len(incoming) >= 10:
            if incoming["timestamp"].max() - incoming["timestamp"].min() <= window:
                members = set(incoming["sender_id"]) | {acc}
                add_ring(members, "smurfing_fan_in")

        if len(outgoing) >= 10:
            if outgoing["timestamp"].max() - outgoing["timestamp"].min() <= window:
                members = set(outgoing["receiver_id"]) | {acc}
                add_ring(members, "smurfing_fan_out")

    # =========================
    # 3️⃣ LAYERED SHELL NETWORKS
    # =========================
    tx_counts = pd.concat([df["sender_id"], df["receiver_id"]]).value_counts()
    shell_nodes = set(tx_counts[(tx_counts >= 2) & (tx_counts <= 3)].index)

    for src in G.nodes():
        for dst in G.nodes():
            if src == dst:
                continue
            try:
                for path in nx.all_simple_paths(G, src, dst, cutoff=5):
                    if len(path) >= 4:
                        mids = path[1:-1]
                        if all(m in shell_nodes for m in mids):
                            add_ring(path, "layered_shell_laundering")
            except:
                pass

    # =========================
    # VISUALIZATION
    # =========================
    net = Network(height="600px", directed=True)

    for node in G.nodes():
        color = "red" if node in suspicious_accounts else "lightblue"
        net.add_node(node, label=node, color=color)

    for s, r in G.edges():
        net.add_edge(s, r)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(temp.name)

    st.subheader("🔍 Transaction Network")
    st.components.v1.html(open(temp.name).read(), height=600)

    # =========================
    # FRAUD RING TABLE
    # =========================
    st.subheader("🚨 Fraud Rings Detected")

    if fraud_rings:
        rings_df = pd.DataFrame(fraud_rings)
        st.dataframe(rings_df, use_container_width=True)
    else:
        st.write("No fraud rings detected")

    # =========================
    # JSON OUTPUT
    # =========================
    output = {
        "fraud_rings": fraud_rings,
        "summary": {
            "total_accounts_analyzed": G.number_of_nodes(),
            "fraud_rings_detected": len(fraud_rings),
            "suspicious_accounts_flagged": len(suspicious_accounts),
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
    }

    st.download_button(
        "Download JSON File",
        data=json.dumps(output, indent=2),
        file_name="money_muling_results.json",
        mime="application/json"
    )

else:
    st.info("Upload a CSV file to begin analysis")
