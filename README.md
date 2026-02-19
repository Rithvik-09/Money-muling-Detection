# 💸 Money Muling Detection System  
### Graph-Based Financial Crime Detection Engine

A web-based financial fraud detection system built using **Python, Streamlit, and NetworkX** to identify money muling rings from transaction data using graph theory.

This project was developed for the **RIFT 2026 Hackathon – Graph Theory / Financial Crime Detection Track**.

---

## 🚀 Features

- 📂 CSV Transaction Upload
- 🔗 Directed Graph Construction
- 🔁 Cycle Detection (3–5 length loops)
- 📥 Fan-In (Smurfing Aggregation) Detection
- 📤 Fan-Out (Smurfing Distribution) Detection
- 🚨 Suspicious Account Highlighting
- 📊 Interactive Graph Visualization
- 📥 Downloadable JSON Fraud Report

---

## 🧠 Problem Statement

Money muling involves transferring illegal funds through multiple accounts to obscure the origin of money.

Traditional database queries fail to detect:
- Multi-hop transaction chains
- Circular routing patterns
- Smurfing (many small transactions)
- Layered transaction structures

This system converts financial transactions into a **directed graph** and applies graph algorithms to detect suspicious structures.

---

## 🏗 System Architecture

```
CSV Upload
   ↓
Data Processing (Pandas)
   ↓
Directed Graph Creation (NetworkX)
   ↓
Fraud Pattern Detection
   ↓
Suspicion Scoring Engine
   ↓
Interactive Visualization (PyVis)
   ↓
JSON Report Generation
```

---

## ⚙️ Tech Stack

- **Python**
- **Streamlit**
- **NetworkX**
- **PyVis**
- **Pandas**

---

## 📁 Project Structure

```
money-muling-detection/
│
├── app.py              # Main Streamlit app + all detection logic
├── requirements.txt    # Python dependencies
├── sample_data.csv     # Sample transaction CSV for testing
├── README.md
└── .gitignore
```

---

## 🔍 Detection Algorithms

### 1️⃣ Cycle Detection
Detects circular fund routing (A → B → C → A)  
Using `networkx.simple_cycles()`  
Cycle length filtered between 3 and 5.

### 2️⃣ Smurfing Detection

**Fan-In Pattern**
- Many accounts sending funds to one account
- Detected using high in-degree

**Fan-Out Pattern**
- One account sending to many accounts
- Detected using high out-degree

---

## 📈 Suspicion Scoring Methodology

Each account is assigned a score (0–100) based on:

- Cycle involvement
- In-degree (incoming transactions)
- Out-degree (outgoing transactions)

Simplified formula:

```
Score = Base + (InDegree × Weight) + (OutDegree × Weight) + Cycle Bonus
```

Score is capped at 100.

---

## 📥 JSON Output Format

```
[
  {
    "account_id": "A",
    "suspicion_score": 85,
    "detected_patterns": ["cycle", "smurfing"],
    "ring_id": "RING_001"
  }
]
```

---

## ▶️ How to Run Locally

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the application

```
streamlit run app.py
```

---

## 📄 Expected CSV Format

```
transaction_id,sender_id,receiver_id,amount,timestamp
T1,A,B,500,2025-02-01 10:00:00
T2,B,C,300,2025-02-01 11:00:00
T3,C,A,200,2025-02-01 12:00:00
```

A ready-to-use `sample_data.csv` is included in the repository for quick testing.

---

## ⚠️ Known Limitations

- Threshold-based smurfing detection
- No merchant false-positive filtering yet
- Limited temporal analysis
- Can be extended with ML-based scoring

---

## 🚀 Future Improvements

- Machine learning fraud classification
- False positive reduction model (merchant filtering)
- Large dataset performance optimization
- Real-time streaming transaction support

---

## 👨‍💻 Author

**Rithvik Gouru**  
GitHub: https://github.com/Rithvik-09

---

## 📜 License

This project is open-source and available under the MIT License.
