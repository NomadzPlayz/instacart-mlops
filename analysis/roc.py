import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from joblib import load

# Paths
FIG_DIR = Path("analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("models/logreg.joblib")
DATA_PATH = Path("data/processed/features.parquet")

# Load data
df = pd.read_parquet(DATA_PATH)

X = df.drop(columns=["user_id", "product_id", "label"])
y = df["label"]

# Load model
model = load(MODEL_PATH)

# Predict probabilities
y_prob = model.predict_proba(X)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y, y_prob)
auc = roc_auc_score(y, y_prob)

# Plot
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Reorder Prediction Model")
plt.legend()
plt.tight_layout()

# Save
plt.savefig(FIG_DIR / "roc_curve.png")
plt.close()

print(f"ROC curve saved (AUC = {auc:.3f})")