import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

st.set_page_config(page_title="Instacart Reorder Predictor", layout="centered")

st.title("🛒 Instacart Reorder Prediction")
st.write(
    "Predict whether a user will reorder a product based on historical behavior."
)

# ================================
# LOAD MODEL FROM MLFLOW
# ================================
@st.cache_resource
def load_model():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

model = load_model()

# ================================
# USER INPUT
# ================================
st.subheader("Input Features")

user_orders = st.number_input(
    "Total orders by user",
    min_value=1,
    max_value=500,
    value=10,
)

avg_days_between = st.number_input(
    "Average days between orders",
    min_value=0.0,
    max_value=60.0,
    value=7.0,
)

product_orders = st.number_input(
    "Total orders for product",
    min_value=1,
    max_value=100_000,
    value=1000,
)

# ================================
# PREDICTION
# ================================
if st.button("Predict Reorder Probability"):
    X = pd.DataFrame(
        [{
            "user_orders": user_orders,
            "avg_days_between": avg_days_between,
            "product_orders": product_orders,
        }]
    )

    prob = model.predict_proba(X)[0][1]

    st.metric(
        label="Reorder Probability",
        value=f"{prob:.2%}",
    )

    if prob > 0.5:
        st.success("High chance of reorder")
    else:
        st.info("Low chance of reorder")
