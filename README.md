# Instacart Reorder Prediction (End-to-End MLOps)

End-to-end machine learning pipeline to predict whether a user will reorder a product on their next purchase, built using the Instacart Market Basket Analysis dataset.

This project demonstrates a **production-style ML workflow** with proper data processing, feature engineering, model training, experiment tracking, and interactive inference via Streamlit.

---

## Problem Statement

Given historical user–product interactions, predict whether a product will be reordered by a user in a future order.

This is a **binary classification problem** with strong temporal constraints and a high risk of data leakage if not handled correctly.

---

## Dataset

Source: **Instacart Market Basket Analysis**

Raw files used:
- `orders.csv`
- `order_products_prior.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

Scale:
- ~3.2M orders  
- ~32M user–product interactions  

---

## Project Structure

```text
instacart-mlops/
├── data/
│   ├── raw/                # Original CSV files
│   └── processed/          # Parquet files (base + features)
│
├── src/
│   ├── ingest.py           # Load & merge raw CSVs → base.parquet
│   ├── check_data.py       # Sanity check raw data
│   ├── features.py         # Feature engineering + label creation
│   └── train.py            # Model training + MLflow tracking
│
├── streamlit_app/
│   └── app.py              # Interactive prediction UI
│
├── models/                 # (Optional) exported models
├── requirements.txt
├── README.md
└── .gitignore

---

## Key Design Decisions

### 1. Temporal Safety (No Data Leakage)

- Data is explicitly sorted by `user_id` and `order_number`
- Labels are generated using future orders via `shift(-1)`
- No future-derived statistics are used as input features
- Product reorder rate is intentionally excluded to avoid leakage

This ensures the model only learns from information that would be available at prediction time.

---

## Feature Engineering

Features are built at three different levels:

### User-level
- `user_orders`: total number of orders made by the user
- `avg_days_between`: average days between consecutive orders

### Product-level
- `product_orders`: total number of times the product was ordered globally

### User–Product Interaction
- `times_bought`: how many times the user bought the product
- `label`: whether the product is reordered in a future order

All features are stored in **Parquet** format for scalability.

---

## Model

- Algorithm: **Logistic Regression**
- Training sample size: **500,000 rows** (randomly sampled for efficiency)
- Solver: `lbfgs`
- Evaluation metric: **ROC AUC**

Final performance:
ROC AUC ≈ 0.63


This score reflects a **leakage-free baseline** rather than an inflated benchmark.

---

## Experiment Tracking (MLflow)

- Tracks metrics (ROC AUC) per training run
- Logs trained model artifacts
- Uses a local SQLite backend (`mlflow.db`)
- Local MLflow artifacts are excluded from version control

This setup mirrors real-world experimentation workflows.

---

## Streamlit App

An interactive Streamlit application is provided for inference and demonstration purposes.

Features:
- Manual feature input
- Real-time reorder probability prediction
- Clean UI for non-technical stakeholders

Run locally:
```bash
streamlit run streamlit_app/app.py

How to Run End-to-End: 
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest raw data
python src/ingest.py

# 4. Build features
python src/features.py

# 5. Train model
python src/train.py

# 6. Launch Streamlit app
streamlit run streamlit_app/app.py

Tech Stack:
- Python 3.12
- Pandas / NumPy
- Scikit-learn
- MLflow
- Streamlit
- Parquet (PyArrow)

What This Project Demonstrates: 
- End-to-end ML pipeline design
- Leakage-aware feature engineering
- Scalable data processing with Parquet
- Experiment tracking with MLflow
- Reproducible training workflow
- Deployment-ready inference interface