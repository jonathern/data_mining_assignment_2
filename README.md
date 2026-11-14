# Data Mining Assignment: Customer Segmentation & Fraud Detection

## Student Information
Name: Jonathan Stephen Kateega
Registration Number: S25M19/006
Access Number: B34976
Course: MSc Data Science & Analytics
Institution: Uganda Christian University (UCU)
Date: 14 November 2026

## Repository Link
[GitHub](https://github.com/jonathern/data_mining_assignment_2)

## Project Overview
This repository was developed as part of a project-based assessment evaluating the integration of practical data science skills, critical thinking, and professional competencies. The notebooks are structured to explicitly address the assessment rubric, specifically focusing on:
- Problem formulation and methodological rigor (justified data prep and algorithm selection).
- Critical evaluation and robustness of findings (comparative metrics, baselines, and sensitivity analysis).
- Reproducibility, communication, and responsible application (executable code, clear visualizations, and proportionate business recommendations).


## Project 1: Wholesale Customer Segmentation

**Objective:** Segment 440 wholesale customers based on annual spending behavior across diverse product categories and recommend tailored service strategies for each segment.

### Key Methodologies & Highlights
* **Data Preparation:** Applied `log1p` transformation to handle severe right-skewness and extreme outliers, followed by Z-score standardization.
* **Cluster Tendency:** Validated the presence of inherent clusters using the Hopkins Statistic (H ≈ 0.78).
* **Algorithm Comparison:** Rigorously compared Partitioning (K-Means), Hierarchical (Agglomerative), and Density-based (DBSCAN) clustering using Silhouette Scores and Davies-Bouldin Index.
* **Business Profiling:** Cross-tabulated clusters with nominal variables (`Channel`, `Region`) to assign descriptive business personas (e.g., *"Fresh & Frozen Foodservice Providers"*, *"Retail Staples Supermarkets"*).

---

## Project 2: Credit Card Fraud Detection

**Objective:** Build an anomaly detector to prioritize transactions for fraud investigation and evaluate its ranking performance against withheld fraud labels in a highly imbalanced dataset (`0.172%` fraud rate).

### Key Methodologies & Highlights
* **Data Preparation:** Utilized `RobustScaler` for the `Amount` feature to prevent extreme outliers from skewing the scaling process. Employed stratified train-test splitting to preserve the `0.172%` fraud ratio.
* **Baseline Establishment:** Established a naive statistical outlier baseline (Z-score > 3) to benchmark machine learning performance.
* **Approach Comparison:** Evaluated Unsupervised Anomaly Detection (Isolation Forest) vs. Supervised Classification (Random Forest with class-weight balancing).
* **Imbalanced Metrics:** Bypassed misleading accuracy metrics in favor of **Precision-Recall AUC (PR-AUC)** and **Recall@K** (top 500 flagged transactions) to simulate real-world investigative constraints.
* **Responsible AI:** Analyzed the critical business distinction between statistical anomalies and malicious fraud to mitigate false-positive costs.

---

## Repository Structure

```text
data_mining_assignment_2/
│
├── .venv/                                      # Virtual environment (ignored by git)
├── data/
│   ├── credit card fraud detection/
│   │   ├── creditcard-README.txt
│   │   └── creditcard.csv                      # Large CSV file >150MB (ignored by git)
│   └── wholesale customers/
│       ├── wholesale-customers-README.txt
│       └── wholesale-customers.csv
├── notebooks/
│   ├── 1. Wholesale customers.ipynb
│   └── 2. Credit Card Fraud Detection.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Data Sources

Due to file size limits, the datasets are not committed directly to this repository. You can download them from the following sources:

* **Wholesale Customers Dataset:**
  * **Source:** UCI Machine Learning Repository
  * **Link:** [Wholesale Customers Dataset](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)
* **Credit Card Fraud Detection Dataset:**
  * **Source:** Kaggle / Machine Learning Group (ULB)
  * **Link:** [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> **Note:** Place the downloaded `.csv` files inside the `data/` directory before running the notebooks.

---

## Setup and Installation

To reproduce the analysis, you will need **Python 3.8+** and the libraries in the `packages.txt`. You can install them via `pip`:

```bash
pip install -r requirements.txt
```

Alternatively, if you use **Conda**:

```bash
conda create -n ds-portfolio python=3.9
conda activate ds-portfolio
conda install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
```

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jonathern/data_mining_assignment_2.git
   cd data_mining_assignment_2
   ```

2. **Prepare the data:**
   Ensure the `data/` folder contains `wholesale-customers.csv` and `creditcard.csv`. *(Note: You may need to update the file paths in the first cell of each notebook if you place the CSVs in the root directory instead of the `data/` folder).*

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Execute the Notebooks:**
   Open either `1. Wholesale customers.ipynb` or `2. Credit Card Fraud Detection.ipynb` and run all cells sequentially (**Kernel -> Restart & Run All**).
