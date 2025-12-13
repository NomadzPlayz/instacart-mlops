import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import dump

DATA = Path("data/processed/features.parquet")

def main():
    print("Loading features...")
    df = pd.read_parquet(DATA)

    df = df.sample(n=500_000, random_state=42)
    print("Sampled shape:", df.shape)

    TARGET = "label"

    X = df.drop(columns=["user_id", "product_id", TARGET])
    y = df[TARGET]
    groups = df["user_id"]

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print("ROC AUC:", auc)

    # 🔥 SAVE MODEL FOR STREAMLIT
    Path("models").mkdir(exist_ok=True)
    dump(model, "models/logreg.joblib")
    print("Model saved to models/logreg.joblib")

if __name__ == "__main__":
    main()
