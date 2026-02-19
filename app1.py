import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import json
import time
from datetime import timedelta
from collections import defaultdict

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="RIFT Money Muling Detector", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #f0f0f0; }
    .explain-box {
        background-color: #1a1a2e; border-left: 4px solid #ff4444;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 10px; color: #f0f0f0;
    }
    .ring-box {
        background-color: #1e1a2e; border-left: 4px solid #ffaa00;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 10px; color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)
st.title("💸 Money Muling Detection Engine")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    start_time = time.time()
    df = pd.read_csv(uploaded_file)

    required = {"transaction_id", "sender_id", "receiver_id", "amount", "timestamp"}
    if not required.issubset(df.columns):
        st.error(f"CSV missing columns. Required: {required}")
        st.stop()

    df["timestamp"]  = pd.to_datetime(df["timestamp"])
    df["sender_id"]   = df["sender_id"].astype(str)
    df["receiver_id"] = df["receiver_id"].astype(str)

    # ── Build directed graph ──────────────────────────────────────────────────
    G = nx.DiGraph()
    for _, r in df.iterrows():
        if G.has_edge(r["sender_id"], r["receiver_id"]):
            G[r["sender_id"]][r["receiver_id"]]["amount"] += r["amount"]
            G[r["sender_id"]][r["receiver_id"]]["count"]  += 1
        else:
            G.add_edge(r["sender_id"], r["receiver_id"],
                       amount=r["amount"], count=1, timestamp=r["timestamp"])

    # ── False-positive guard ──────────────────────────────────────────────────
    tx_counts   = pd.concat([df["sender_id"], df["receiver_id"]]).value_counts()
    counterpart = defaultdict(set)
    for _, r in df.iterrows():
        counterpart[r["sender_id"]].add(r["receiver_id"])
        counterpart[r["receiver_id"]].add(r["sender_id"])

    MERCHANT_THRESHOLD = 50
    legitimate_hubs = {
        acc for acc, partners in counterpart.items()
        if len(partners) >= MERCHANT_THRESHOLD
    }

    fraud_rings       = []
    account_scores    = defaultdict(float)
    account_patterns  = defaultdict(set)
    account_ring_id   = {}
    account_reasons   = defaultdict(list)   # NEW: human-readable why
    account_evidence  = defaultdict(list)   # NEW: triggering transactions
    ring_explanations = {}                  # NEW: how each ring formed

    ring_counter = [0]

    def next_ring_id():
        ring_counter[0] += 1
        return f"RING_{ring_counter[0]:03d}"

    def flag_accounts(members, score, pattern, ring_id, reason_fn):
        for acc in members:
            if acc in legitimate_hubs:
                continue
            account_scores[acc] = min(100.0, account_scores[acc] + score * 0.5
                                      if account_scores[acc] > 0 else score)
            account_patterns[acc].add(pattern)
            if acc not in account_ring_id:
                account_ring_id[acc] = ring_id
            reason = reason_fn(acc)
            if reason not in account_reasons[acc]:
                account_reasons[acc].append(reason)

    def add_evidence(acc_id, txn_rows):
        for _, row in txn_rows.iterrows():
            entry = {
                "transaction_id": row["transaction_id"],
                "sender":         row["sender_id"],
                "receiver":       row["receiver_id"],
                "amount":         round(float(row["amount"]), 2),
                "timestamp":      str(row["timestamp"]),
            }
            if entry not in account_evidence[acc_id]:
                account_evidence[acc_id].append(entry)

    # ══════════════════════════════════════════════════════════════════════════
    # PATTERN 1 — CIRCULAR FLOWS
    # ══════════════════════════════════════════════════════════════════════════
    for cycle in nx.simple_cycles(G):
        if 3 <= len(cycle) <= 5:
            if all(n in legitimate_hubs for n in cycle):
                continue
            ring_id    = next_ring_id()
            n          = len(cycle)
            cycle_path = " → ".join(cycle) + " → " + cycle[0]

            fraud_rings.append({
                "ring_id": ring_id, "member_accounts": list(cycle),
                "pattern_type": "cycle", "risk_score": 95.3, "member_count": n,
            })

            ring_explanations[ring_id] = (
                f"**Circular Fund Routing — {n}-node cycle**\n\n"
                f"Money flows in a closed loop: `{cycle_path}`\n\n"
                f"Funds originate from one account, pass through {n-1} intermediate(s), "
                f"and return to the start — a classic technique to disguise the origin of illicit money."
            )

            def make_cycle_reason(acc, cycle=cycle, cycle_path=cycle_path, n=n):
                pos = cycle.index(acc) if acc in cycle else -1
                if pos == 0:
                    role = "the **originator** — starts and receives the money back"
                elif pos == n - 1:
                    role = "the **final relay** — last hop before money returns to origin"
                else:
                    role = f"an **intermediate relay** — hop {pos+1} of {n}"
                return (f"Part of {n}-node cycle `{cycle_path}`. "
                        f"This account is {role}.")

            flag_accounts(cycle, 95.3, f"cycle_length_{n}", ring_id,
                          reason_fn=make_cycle_reason)

            for i in range(len(cycle)):
                s = cycle[i]; d = cycle[(i+1) % len(cycle)]
                txns = df[(df["sender_id"] == s) & (df["receiver_id"] == d)]
                add_evidence(s, txns); add_evidence(d, txns)

    # ══════════════════════════════════════════════════════════════════════════
    # PATTERN 2 — SMURFING
    # ══════════════════════════════════════════════════════════════════════════
    SMUF_THRESHOLD = 10
    WINDOW         = timedelta(hours=72)
    seen_smrf      = set()

    for acc in list(G.nodes()):
        if acc in legitimate_hubs:
            continue

        # Fan-in
        inc            = df[df["receiver_id"] == acc]
        unique_senders = inc["sender_id"].nunique()
        if (unique_senders >= SMUF_THRESHOLD
                and (inc["timestamp"].max() - inc["timestamp"].min()) <= WINDOW
                and f"in_{acc}" not in seen_smrf):
            seen_smrf.add(f"in_{acc}")
            ring_id   = next_ring_id()
            members   = list(set(inc["sender_id"].tolist()) | {acc})
            total_in  = round(inc["amount"].sum(), 2)
            span_hrs  = round((inc["timestamp"].max() - inc["timestamp"].min()).total_seconds()/3600, 1)

            fraud_rings.append({
                "ring_id": ring_id, "member_accounts": members,
                "pattern_type": "smurfing_fan_in", "risk_score": 85.0, "member_count": len(members),
            })
            ring_explanations[ring_id] = (
                f"**Smurfing — Fan-in (Aggregation)**\n\n"
                f"Account `{acc}` received from **{unique_senders} unique senders** "
                f"within **{span_hrs} hours**, totalling **{total_in}** units.\n\n"
                f"Many mules deposit small amounts into one aggregator to pool funds "
                f"below reporting thresholds. `{acc}` is the aggregator."
            )

            def fin_reason(a, acc=acc, unique_senders=unique_senders,
                           span_hrs=span_hrs, total_in=total_in, inc=inc):
                if a == acc:
                    return (f"**Aggregator** in fan-in smurfing. Received from "
                            f"{unique_senders} senders in {span_hrs}h — total {total_in} units.")
                amt = round(inc[inc["sender_id"]==a]["amount"].sum(), 2)
                return (f"**Mule/sender** in fan-in smurfing. Sent {amt} units to "
                        f"aggregator `{acc}` alongside {unique_senders-1} other senders in {span_hrs}h.")

            flag_accounts(members, 85.0, "high_velocity", ring_id, reason_fn=fin_reason)
            add_evidence(acc, inc)
            for sender in inc["sender_id"].unique():
                add_evidence(sender, inc[inc["sender_id"]==sender])

        # Fan-out
        out              = df[df["sender_id"] == acc]
        unique_receivers = out["receiver_id"].nunique()
        if (unique_receivers >= SMUF_THRESHOLD
                and (out["timestamp"].max() - out["timestamp"].min()) <= WINDOW
                and f"out_{acc}" not in seen_smrf):
            seen_smrf.add(f"out_{acc}")
            ring_id   = next_ring_id()
            members   = list(set(out["receiver_id"].tolist()) | {acc})
            total_out = round(out["amount"].sum(), 2)
            span_hrs  = round((out["timestamp"].max() - out["timestamp"].min()).total_seconds()/3600, 1)

            fraud_rings.append({
                "ring_id": ring_id, "member_accounts": members,
                "pattern_type": "smurfing_fan_out", "risk_score": 85.0, "member_count": len(members),
            })
            ring_explanations[ring_id] = (
                f"**Smurfing — Fan-out (Dispersal)**\n\n"
                f"Account `{acc}` sent to **{unique_receivers} unique receivers** "
                f"within **{span_hrs} hours**, totalling **{total_out}** units.\n\n"
                f"One account rapidly disperses funds to many recipients to break the audit trail. "
                f"`{acc}` is the disperser; the {unique_receivers} receivers are end mules."
            )

            def fout_reason(a, acc=acc, unique_receivers=unique_receivers,
                            span_hrs=span_hrs, total_out=total_out, out=out):
                if a == acc:
                    return (f"**Disperser** in fan-out smurfing. Sent to "
                            f"{unique_receivers} receivers in {span_hrs}h — total {total_out} units.")
                amt = round(out[out["receiver_id"]==a]["amount"].sum(), 2)
                return (f"**Receiver/mule** in fan-out smurfing. Received {amt} units from "
                        f"disperser `{acc}` alongside {unique_receivers-1} others in {span_hrs}h.")

            flag_accounts(members, 85.0, "high_velocity", ring_id, reason_fn=fout_reason)
            add_evidence(acc, out)
            for recv in out["receiver_id"].unique():
                add_evidence(recv, out[out["receiver_id"]==recv])

    # ══════════════════════════════════════════════════════════════════════════
    # PATTERN 3 — SHELL LAYERING
    # ══════════════════════════════════════════════════════════════════════════
    shell_nodes = set(
        tx_counts[(tx_counts >= 2) & (tx_counts <= 3)].index.astype(str)
    ) - legitimate_hubs

    seen_shell_paths = set()
    shell_sources    = {s for s, d in G.edges() if d in shell_nodes}

    for src in shell_sources:
        if src in legitimate_hubs:
            continue
        stack = [(src, [src])]
        while stack:
            node, path = stack.pop()
            if len(path) > 5:
                continue
            for nbr in G.successors(node):
                new_path = path + [nbr]
                if len(new_path) >= 4:
                    mids = new_path[1:-1]
                    if all(m in shell_nodes for m in mids):
                        key = tuple(new_path)
                        if key not in seen_shell_paths:
                            seen_shell_paths.add(key)
                            ring_id  = next_ring_id()
                            n_hops   = len(new_path) - 1
                            path_str = " → ".join(new_path)
                            shell_str = ", ".join(mids)

                            fraud_rings.append({
                                "ring_id": ring_id, "member_accounts": list(new_path),
                                "pattern_type": "shell_layering", "risk_score": 90.0,
                                "member_count": len(new_path),
                            })
                            ring_explanations[ring_id] = (
                                f"**Layered Shell Network — {n_hops}-hop chain**\n\n"
                                f"Money flows: `{path_str}`\n\n"
                                f"Intermediate account(s) `{shell_str}` are **shell accounts** "
                                f"(only 2–3 total transactions) — throwaway accounts created to "
                                f"obscure the trail between `{new_path[0]}` (source) and "
                                f"`{new_path[-1]}` (destination)."
                            )

                            def shell_reason(a, path=new_path, mids=mids, path_str=path_str, n_hops=n_hops):
                                tx_c = int(tx_counts.get(a, 0))
                                if a == path[0]:
                                    return (f"**Source** of {n_hops}-hop shell chain `{path_str}`. "
                                            f"Initiates fund flow through throwaway shell accounts.")
                                elif a == path[-1]:
                                    return (f"**Destination** of {n_hops}-hop shell chain `{path_str}`. "
                                            f"Receives layered funds after passing through shells.")
                                else:
                                    return (f"**Shell account** in `{path_str}`. "
                                            f"Only {tx_c} total transactions — throwaway layering account.")

                            flag_accounts(new_path, 90.0, "layered_chain", ring_id,
                                          reason_fn=shell_reason)

                            for i in range(len(new_path)-1):
                                s_n = new_path[i]; d_n = new_path[i+1]
                                txns = df[(df["sender_id"]==s_n) & (df["receiver_id"]==d_n)]
                                add_evidence(s_n, txns); add_evidence(d_n, txns)

                if nbr in shell_nodes and len(new_path) < 5:
                    stack.append((nbr, new_path))

    # ── Build suspicious_accounts list ───────────────────────────────────────
    suspicious_list = []
    for acc_id in account_scores:
        suspicious_list.append({
            "account_id":        acc_id,
            "suspicion_score":   round(account_scores[acc_id], 2),
            "detected_patterns": sorted(account_patterns[acc_id]),
            "ring_id":           account_ring_id.get(acc_id, "UNKNOWN"),
        })
    suspicious_list.sort(key=lambda x: x["suspicion_score"], reverse=True)
    suspicious_ids = set(account_scores.keys())

    # ══════════════════════════════════════════════════════════════════════════
    # UI — METRICS
    # ══════════════════════════════════════════════════════════════════════════
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Accounts",      G.number_of_nodes())
    col2.metric("Suspicious Accounts", len(suspicious_list))
    col3.metric("Fraud Rings",         len(fraud_rings))
    col4.metric("Processing Time",     f"{round(time.time()-start_time,2)}s")

    # ══════════════════════════════════════════════════════════════════════════
    # UI — VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    net = Network(height="600px", width="100%", directed=True,
                  bgcolor="#1a1a2e", font_color="white")
    net.set_options(json.dumps({
        "nodes": {"font": {"size": 12}},
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                  "color": {"color": "#888888"}},
        "interaction": {"hover": True, "tooltipDelay": 100},
        "physics": {"stabilization": {"iterations": 100}}
    }))

    for node in G.nodes():
        is_susp = node in suspicious_ids
        score   = round(account_scores.get(node, 0), 1)
        patterns = list(account_patterns.get(node, []))
        reasons  = account_reasons.get(node, [])
        if is_susp:
            tip = (f"<b>{node}</b><br>Score: {score}<br>"
                   f"Patterns: {', '.join(patterns)}<br>"
                   f"Reason: {reasons[0][:100] if reasons else 'N/A'}")
        else:
            tip = f"<b>{node}</b><br>Clean account"
        color = "#ff4444" if is_susp else "#4488ff"
        size  = 30 if is_susp else 15
        net.add_node(node, label=str(node), color=color, size=size, title=tip)

    for s, r, data in G.edges(data=True):
        net.add_edge(s, r, title=f"Amount: {data['amount']:.2f}")

    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.save_graph(tmp_path)

    st.subheader("🔍 Interactive Network Visualization")
    st.caption("🔴 Red = suspicious  🔵 Blue = clean  |  Hover over nodes/edges for details")
    st.components.v1.html(open(tmp_path).read(), height=620)

    # ══════════════════════════════════════════════════════════════════════════
    # UI — FRAUD RING SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🚨 Fraud Ring Summary")
    if fraud_rings:
        display_df = pd.DataFrame(fraud_rings)
        display_df["member_accounts"] = display_df["member_accounts"].apply(
            lambda x: ", ".join(map(str, x)))
        st.dataframe(
            display_df[["ring_id", "pattern_type", "member_count", "risk_score", "member_accounts"]],
            use_container_width=True)
    else:
        st.info("No fraud rings detected.")

    # ══════════════════════════════════════════════════════════════════════════
    # UI — HOW EACH RING FORMED  ← NEW
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🧩 How Each Fraud Ring Was Formed")
    st.caption("Expand any ring to see exactly what happened, who is involved, and the evidence transactions.")

    ICONS = {"cycle": "🔄", "smurfing_fan_in": "📥",
             "smurfing_fan_out": "📤", "shell_layering": "🐚"}

    for ring in fraud_rings:
        rid     = ring["ring_id"]
        ptype   = ring["pattern_type"]
        rscore  = ring["risk_score"]
        members = ring["member_accounts"]
        explain = ring_explanations.get(rid, "No explanation available.")
        icon    = ICONS.get(ptype, "⚠️")

        with st.expander(f"{icon}  {rid}  |  {ptype}  |  Risk: {rscore}  |  {len(members)} accounts"):
            st.markdown(explain)

            st.markdown("**📋 Transactions that formed this ring:**")
            ring_member_set = set(map(str, members))
            ring_txns = df[
                df["sender_id"].isin(ring_member_set) &
                df["receiver_id"].isin(ring_member_set)
            ][["transaction_id", "sender_id", "receiver_id", "amount", "timestamp"]].copy()
            ring_txns["amount"] = ring_txns["amount"].round(2)

            if not ring_txns.empty:
                st.dataframe(ring_txns.reset_index(drop=True), use_container_width=True)
            else:
                st.info("No internal transactions found between ring members.")

    # ══════════════════════════════════════════════════════════════════════════
    # UI — SUSPICIOUS ACCOUNT DEEP DIVE  ← NEW
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🔎 Suspicious Account Deep Dive")
    st.caption("Select any flagged account to see exactly WHY it was flagged and the evidence.")

    if suspicious_list:
        acc_options = [f"{s['account_id']}  (Score: {s['suspicion_score']})"
                       for s in suspicious_list]
        selected     = st.selectbox("Choose an account to investigate:", acc_options)
        selected_acc = selected.split("  ")[0].strip()
        acc_data     = next((s for s in suspicious_list if s["account_id"] == selected_acc), None)

        if acc_data:
            c1, c2, c3 = st.columns(3)
            c1.metric("Suspicion Score", acc_data["suspicion_score"])
            c2.metric("Primary Ring",    acc_data["ring_id"])
            c3.metric("Patterns Found",  len(acc_data["detected_patterns"]))

            # Why suspicious
            st.markdown("**🚩 Why is this account suspicious?**")
            for i, reason in enumerate(account_reasons.get(selected_acc, []), 1):
                st.markdown(
                    f'<div class="explain-box"><b>Reason {i}:</b> {reason}</div>',
                    unsafe_allow_html=True)

            # Pattern explanations
            st.markdown("**🏷️ What each detected pattern means:**")
            pattern_explanations = {
                "cycle_length_3": "In a 3-node circular loop (A→B→C→A). Direct evidence of round-tripping funds.",
                "cycle_length_4": "In a 4-node circular loop. Extra hop added to obscure origin.",
                "cycle_length_5": "In a 5-node circular loop. Maximum complexity cycle.",
                "high_velocity":  "Part of a smurfing network — either aggregating from many senders or dispersing to many receivers in a short time window.",
                "layered_chain":  "Part of a shell layering chain — money passed through low-activity throwaway accounts.",
            }
            for pat in sorted(acc_data["detected_patterns"]):
                expl = pattern_explanations.get(pat, "Suspicious activity pattern.")
                st.markdown(
                    f'<div class="ring-box"><b>{pat}:</b> {expl}</div>',
                    unsafe_allow_html=True)

            # Transaction stats
            st.markdown("**📊 Transaction Statistics:**")
            sent     = df[df["sender_id"]   == selected_acc]
            received = df[df["receiver_id"] == selected_acc]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Sent",          round(sent["amount"].sum(), 2))
            s2.metric("Total Received",       round(received["amount"].sum(), 2))
            s3.metric("Unique Sent-To",       sent["receiver_id"].nunique())
            s4.metric("Unique Received-From", received["sender_id"].nunique())

            # Evidence transactions
            st.markdown("**🧾 Evidence Transactions (triggered the flag):**")
            evidence = account_evidence.get(selected_acc, [])
            if evidence:
                ev_df = pd.DataFrame(evidence).drop_duplicates(subset=["transaction_id"])
                st.dataframe(ev_df, use_container_width=True)
            else:
                st.info("No specific evidence transactions recorded.")

            # All transactions
            st.markdown("**📁 All Transactions for This Account:**")
            all_txns = df[
                (df["sender_id"] == selected_acc) |
                (df["receiver_id"] == selected_acc)
            ][["transaction_id", "sender_id", "receiver_id", "amount", "timestamp"]].copy()
            all_txns["amount"]    = all_txns["amount"].round(2)
            all_txns["direction"] = all_txns.apply(
                lambda r: "📤 SENT" if r["sender_id"] == selected_acc else "📥 RECEIVED", axis=1)
            st.dataframe(all_txns.reset_index(drop=True), use_container_width=True)

            # All rings this account belongs to
            st.markdown("**🔗 All Rings This Account Belongs To:**")
            my_rings = [r for r in fraud_rings if selected_acc in r["member_accounts"]]
            if my_rings:
                for r in my_rings:
                    rid   = r["ring_id"]
                    expl  = ring_explanations.get(rid, "")
                    st.markdown(
                        f'<div class="ring-box">'
                        f'<b>{rid}</b> ({r["pattern_type"]}) — Risk: {r["risk_score"]}<br>'
                        f'Members: {", ".join(map(str, r["member_accounts"]))}<br><br>'
                        f'{expl}'
                        f'</div>', unsafe_allow_html=True)
            else:
                st.info("No direct ring membership found.")

    # ══════════════════════════════════════════════════════════════════════════
    # UI — FULL SUSPICIOUS ACCOUNTS TABLE  ← NEW
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📋 All Suspicious Accounts")
    if suspicious_list:
        full_table = []
        for s in suspicious_list:
            acc = s["account_id"]
            reasons = account_reasons.get(acc, [])
            full_table.append({
                "Account ID":        acc,
                "Suspicion Score":   s["suspicion_score"],
                "Primary Ring":      s["ring_id"],
                "Patterns":          ", ".join(s["detected_patterns"]),
                "Why Suspicious":    reasons[0] if reasons else "—",
                "Total Sent":        round(df[df["sender_id"]   == acc]["amount"].sum(), 2),
                "Total Received":    round(df[df["receiver_id"] == acc]["amount"].sum(), 2),
            })
        st.dataframe(pd.DataFrame(full_table), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # JSON OUTPUT
    # ══════════════════════════════════════════════════════════════════════════
    json_output = {
        "suspicious_accounts": suspicious_list,
        "fraud_rings":         fraud_rings,
        "summary": {
            "total_accounts_analyzed":     G.number_of_nodes(),
            "suspicious_accounts_flagged": len(suspicious_list),
            "fraud_rings_detected":        len(fraud_rings),
            "processing_time_seconds":     round(time.time() - start_time, 3),
        },
    }

    st.download_button(
        label="⬇️ Download Results (JSON)",
        data=json.dumps(json_output, indent=4, default=str),
        file_name="money_muling_results.json",
        mime="application/json",
    )