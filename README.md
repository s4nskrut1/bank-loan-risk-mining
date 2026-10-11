# 🏦 CrediMine — Bank Loan Default & Customer Risk Mining Platform

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat&logo=flask&logoColor=white)](https://palletsprojects.com/p/flask/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-v3.0-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

A comprehensive, full-stack **Data Mining & Risk Intelligence Platform** developed for banking and credit risk assessment. **CrediMine** mines historical loan records to discover high-risk default patterns, identifies attributes with the strongest predictive and statistical relationship with loan default, extracts auditable decision-tree IF-THEN rules, and provides a real-time risk assessment simulator for credit underwriters.

---

## 📑 Core Academic Syllabus Alignment

| Academic Concept | Syllabus Mapping | Implementation in CrediMine |
|---|---|---|
| **Data Cleaning & Preparation** | Data Preprocessing | Automated missing value imputation (median/mode), categorical label encoding, feature scaling, and Loan-to-Income Ratio (LIR) derivation. |
| **Attribute Relevance Analysis** | Concept Description & Characterization | **Resolves Core Student Question**: Synthesizes **Mutual Information (Information Gain)**, **Spearman Rank Correlation ($r$)**, and **Tree Gini Split Importance** into a unified ranking. |
| **Statistical-Based Classification** | Bayesian Learning | **Gaussian Naive Bayes** calculating posterior class probabilities under conditional independence. |
| **Decision-Based Classification** | Decision Trees | **CART Decision Tree** with programmatic IF-THEN rule extraction and node purity tracking. |
| **Ensemble Benchmark** | Advanced Data Mining | **Random Forest Classifier** (100 estimators) mitigating variance and overfitting. |
| **Dynamic Ingestion** | Modern Production Pipeline | Ingest any `.csv` dataset dynamically, validate required attributes, retrain visual models on the fly, and render live dashboards. |

---

## 🎯 Resolving the Faculty Core Question:
> *"Which attributes have the strongest relationship with loan default?"*

CrediMine resolves this question by combining three mathematical perspectives:
1. **Mutual Information ($I(X; Y)$)**: Detects non-linear statistical dependencies without parametric assumptions.
2. **Spearman Rank Correlation ($r$)**: Measures monotonic directional association with default records.
3. **Random Forest Gini Impurity Importance**: Measures practical split contribution within tree ensembles.

### **Key Mining Insights:**
- **Tier 1 — Dominant Drivers**:
  1. **Repayment History** ($I(X;Y) \approx 0.145$, strong negative monotonic association): Prior delinquency is the single strongest indicator; customers with frequent missed payments exhibit an $>70\%$ historical default rate.
  2. **Credit History Bureau Standing** ($I(X;Y) \approx 0.103$): Serves as the primary external creditworthiness anchor.
- **Tier 2 — High Financial Stress**:
  3. **Loan-to-Income Ratio (LIR)**: Measures leverage stress ($\text{Loan Amount} / \text{Annual Income}$). Ratios exceeding $1.5\times$ dramatically elevate default probability.
  4. **Existing Loans**: High active debt count reflects credit over-extension.
- **Tier 3 — Demographic / Mitigating Factors**:
  - **Age** and **Loan Amount (in isolation)** exhibit lower direct correlation, demonstrating that credit behavior and leverage outweigh raw demographic indicators.
  - **Account Balance** and **Employment Duration** act as mitigating risk buffers.

---

## 🚀 3-Stage User Journey

1. **Clean Landing Page**: Introduction to CrediMine with quick navigation and direct explore button (no cluttered forms on home).
2. **Dedicated Dataset Ingestion Screen**: Drag-and-drop or upload any CSV, view required attribute checklist, or load benchmark records with one click.
3. **Comprehensive Mining Dashboard**:
   - **Overview & Statistical EDA**: Key KPIs, risk distribution donut chart, income vs. loan amount scatter plot, credit history default rate breakdown, and raw dataset preview.
   - **Attribute Relevance Lab**: Answering the core question with Mutual Information bar chart, Tree Gini Importance chart, correlation matrix heatmap, and ranked composite table.
   - **Classification Models**: Performance comparison (Accuracy, Precision, Recall, F1-Score, Training Time) and interactive Confusion Matrices for Decision Tree, Naive Bayes, and Random Forest.
   - **Decision Tree & Rules**: Visual decision tree node hierarchy and human-readable, auditable IF-THEN production rules.
   - **Live Risk Assessment**: 8-input interactive applicant simulator with instant real-time risk verdict (`Low Risk`, `Medium Risk`, `High Risk`), probability gauge, and contributing factor badges.

---

## 📂 Project Structure

```
bank_loan_risk_mining/
│
├── app.py                         # Flask server & REST API endpoints
├── mining_engine.py               # Core Data Mining, Relevance & scikit-learn models engine
├── generate_data.py               # Basel-aligned benchmark data generator
├── generate_test_datasets.py      # Generator for 4 specialized test banking portfolios
│
├── historical_loan_records.csv    # Default calibrated benchmark dataset (1,200 records)
├── sample_loan_template.csv       # Formatted template CSV for external data upload
│
├── sample_datasets/               # 4 Pre-generated specialized test datasets
│   ├── retail_banking_tier1.csv       # 800 records (Prime metro borrowers, 12.6% default)
│   ├── microfinance_rural_lending.csv # 1,000 records (Semi-urban & SME loans, 31.1% default)
│   ├── fintech_digital_loans.csv      # 1,500 records (Young consumer unsecured credit, 32.3% default)
│   └── stressed_credit_portfolio.csv  # 600 records (High-risk stressed asset pool, 67.7% default)
│
├── templates/
│   └── index.html                 # Full responsive UI (Landing, Upload, Dashboard)
│
├── static/
│   └── images/                    # UI assets and illustrations
│
├── uploads/                       # Directory for uploaded CSV datasets (.gitkeep tracked)
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation & viva guide
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/bank-loan-risk-mining.git
cd bank-loan-risk-mining
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Platform
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 📊 Extracted Decision Rules (Sample)

The platform programmatically extracts explicit, human-auditable rules directly from the fitted Decision Tree:
- `IF Repayment History == On-Time AND DTI Ratio <= 0.80 ➜ THEN Low Risk (Confidence: 98.4%)`
- `IF Repayment History == Frequent Delinquency AND Credit History == Poor ➜ THEN High Risk (Confidence: 100.0%)`
- `IF Repayment History == 1-2 Missed Payments AND DTI Ratio > 1.20 ➜ THEN Medium Risk (Confidence: 82.5%)`

---

## 🎓 Viva Q&A Preparation Cheat Sheet

1. **Q: Why classify into Low / Medium / High Risk instead of binary 0/1 default?**
   - *A: Real-world retail banking follows Basel II / RBI prudential guidelines where accounts are stratified into Standard Performing (Low), Substandard / Watchlist (Medium), and Doubtful / Impaired (High). This enables risk-based pricing, proactive monitoring, and custom loan quantum limits.*
2. **Q: Why compare Naive Bayes against Decision Trees?**
   - *A: Decision Trees partition the feature space orthogonally using Information Gain (Decision-Based classification), whereas Naive Bayes calculates posterior probabilities assuming conditional feature independence (Statistical-Based classification).*
3. **Q: Why is High-Risk Recall prioritized over overall Accuracy?**
   - *A: Missing an actual default (False Negative) causes direct capital write-offs for a lending institution, which is 5–10× more costly than an unnecessary manual audit (False Positive).*

---

## 📜 License
This project is developed for educational and academic mini-project coursework in Data Mining & Machine Learning.
