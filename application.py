import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import json
import time

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Money Muling Detector", layout="wide")

st.title("💸 Money Muling Detection System")
st.write("Upload transaction CSV file to analyze suspicious money flow patterns")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:

    start_time = time.time()

    # Read CSV
    df = pd.read_csv(uploaded_file)

    required_columns = {
        "transaction_id",
        "sender_id",
        "receiver_id",
        "amount",
        "timestamp"
    }

    if not required_columns.issubset(df.columns):
        st.error("CSV must contain columns: transaction_id, sender_id, receiver_id, amount, timestamp")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    st.subheader("📄 Transaction Data")
    st.dataframe(df)

    # -----------------------------
    # Build graph
    # -----------------------------
    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_edge(
            row["sender_id"],
            row["receiver_id"],
            amount=float(row["amount"]),
            timestamp=row["timestamp"]
        )

    # -----------------------------
    # Detect cycles
    # -----------------------------
    cycles = []
    for cycle in nx.simple_cycles(G):
        if 3 <= len(cycle) <= 5:
            cycles.append(cycle)

    suspicious_accounts = {}
    fraud_rings = []

    ring_no = 1

    for cycle in cycles:
        ring_id = f"RING_{ring_no:03d}"

        for acc in cycle:
            suspicious_accounts[acc] = {
                "account_id": acc,
                "suspicion_score": 40.0,
                "detected_patterns": ["cycle"],
                "ring_id": ring_id
            }

        fraud_rings.append({
            "ring_id": ring_id,
            "member_accounts": cycle,
            "pattern_type": "circular_flow",
            "risk_score": 90.0
        })

        ring_no += 1


    # -----------------------------
# Detect High Fan-Out Accounts
# -----------------------------
fanout_threshold = 4  # you can tune this
for node in G.nodes():
    if G.out_degree(node) > fanout_threshold:
        if node not in suspicious_accounts:
            suspicious_accounts[node] = {
                "account_id": node,
                "suspicion_score": 25.0,
                "detected_patterns": ["high_fanout"],
                "ring_id": None
            }
        else:
            suspicious_accounts[node]["suspicion_score"] += 25.0
            suspicious_accounts[node]["detected_patterns"].append("high_fanout")
    


# -----------------------------
# Detect Rapid Multi-Hop Transfers
# -----------------------------
df_sorted = df.sort_values("timestamp")

time_threshold_minutes = 30

for i in range(len(df_sorted) - 2):
    t1 = df_sorted.iloc[i]
    t2 = df_sorted.iloc[i + 1]
    t3 = df_sorted.iloc[i + 2]

    # Check chain pattern
    if t1["receiver_id"] == t2["sender_id"] and \
       t2["receiver_id"] == t3["sender_id"]:

        time_diff = (t3["timestamp"] - t1["timestamp"]).total_seconds() / 60

        if time_diff <= time_threshold_minutes:
            for acc in [t1["sender_id"], t2["sender_id"], t3["receiver_id"]]:
                if acc not in suspicious_accounts:
                    suspicious_accounts[acc] = {
                        "account_id": acc,
                        "suspicion_score": 30.0,
                        "detected_patterns": ["rapid_multi_hop"],
                        "ring_id": None
                    }
                else:
                    suspicious_accounts[acc]["suspicion_score"] += 30.0
                    suspicious_accounts[acc]["detected_patterns"].append("rapid_multi_hop")


# -----------------------------
# Detect Large Transactions
# -----------------------------
amount_threshold = df["amount"].mean() * 3

for _, row in df.iterrows():
    if row["amount"] > amount_threshold:
        acc = row["sender_id"]

        if acc not in suspicious_accounts:
            suspicious_accounts[acc] = {
                "account_id": acc,
                "suspicion_score": 20.0,
                "detected_patterns": ["large_transaction"],
                "ring_id": None
            }
        else:
            suspicious_accounts[acc]["suspicion_score"] += 20.0
            suspicious_accounts[acc]["detected_patterns"].append("large_transaction")
# -----------------------------
# Detect Large Transactions
# -----------------------------
amount_threshold = df["amount"].mean() * 3

for _, row in df.iterrows():
    if row["amount"] > amount_threshold:
        acc = row["sender_id"]

        if acc not in suspicious_accounts:
            suspicious_accounts[acc] = {
                "account_id": acc,
                "suspicion_score": 20.0,
                "detected_patterns": ["large_transaction"],
                "ring_id": None
            }
        else:
            suspicious_accounts[acc]["suspicion_score"] += 20.0
            suspicious_accounts[acc]["detected_patterns"].append("large_transaction")

    # -----------------------------
    # Visualize graph
    # -----------------------------
    net = Network(height="600px", directed=True)

    for node in G.nodes():
        if node in suspicious_accounts:
            net.add_node(node, label=node, color="red")
        else:
            net.add_node(node, label=node)

    for src, dst in G.edges():
        net.add_edge(src, dst)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(temp_file.name)

    st.subheader("🔍 Transaction Network")
    st.components.v1.html(open(temp_file.name).read(), height=600)

    # -----------------------------
    # Fraud summary
    # -----------------------------
    st.subheader("🚨 Fraud Rings Detected")

    if fraud_rings:
        st.table(pd.DataFrame(fraud_rings))
    else:
        st.write("No fraud rings detected")


for acc in suspicious_accounts:
    score = suspicious_accounts[acc]["suspicion_score"]

    if score >= 70:
        suspicious_accounts[acc]["risk_level"] = "HIGH"
    elif score >= 40:
        suspicious_accounts[acc]["risk_level"] = "MEDIUM"
    else:
        suspicious_accounts[acc]["risk_level"] = "LOW"

    # -----------------------------
    # JSON output
    # -----------------------------
    process_time = round(time.time() - start_time, 2)

    output = {
        "suspicious_accounts": sorted(
            suspicious_accounts.values(),
            key=lambda x: x["suspicion_score"],
            reverse=True
        ),
        "fraud_rings": fraud_rings,
        "summary": {
            "total_accounts_analyzed": G.number_of_nodes(),
            "suspicious_accounts_flagged": len(suspicious_accounts),
            "fraud_rings_detected": len(fraud_rings),
            "processing_time_seconds": process_time
        }
    }

    st.subheader("📥 Download Result JSON")

    st.download_button(
        "Download JSON File",
        data=json.dumps(output, indent=2),
        file_name="money_muling_output.json",
        mime="application/json"
    )

else:
    st.info("Please upload a CSV file to begin analysis.")