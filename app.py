# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)
import joblib
import matplotlib.pyplot as plt
import altair as alt
import os
import datetime
from xgboost import XGBClassifier
from sklearn.svm import SVC
import datetime

# set histories
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "retraining_history" not in st.session_state:
    st.session_state.retraining_history = []

# configuration
st.set_page_config(
    page_title="Telecom Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Feature Engineering
def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=['State', 'Area code'], errors='ignore')

    # Encode binary categorical variables
    if 'International plan' in df.columns:
        df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
    if 'Voice mail plan' in df.columns:
        df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})

    # Combined totals
    # Compute and log-transform total minutes
    df['Total minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes'] + df['Total intl minutes']
    df['Total minutes'] = np.log1p(df['Total minutes'])

    # Compute and log-transform total charge
    df['Total charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']
    df['Total charge'] = np.log1p(df['Total charge'])

    # Compute and log-transform total calls
    df['Total calls'] = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
    df['Total calls'] = np.log1p(df['Total calls'])


    # Ratios
    df['Day charge share'] = df['Total day charge'] / df['Total charge']
    df['Eve charge share'] = df['Total eve charge'] / df['Total charge']
    df['Night charge share'] = df['Total night charge'] / df['Total charge']
    df['Intl charge share'] = df['Total intl charge'] / df['Total charge']

    df['Day calls share'] = df['Total day calls'] / df['Total calls']
    df['Eve calls share'] = df['Total eve calls'] / df['Total calls']
    df['Night calls share'] = df['Total night calls'] / df['Total calls']
    df['Intl calls share'] = df['Total intl calls'] / df['Total calls']

    df['Minutes per call'] = df['Total minutes'] / df['Total calls']
    df['Charge per minute'] = df['Total charge'] / df['Total minutes']

    df['Weighted service calls'] = df['Customer service calls'] * df['Account length']
    df['Service call share'] = df['Customer service calls'] / df['Total calls']
    df['Weighted service call share'] = df['Service call share'] * df['Account length']

    # Drop unused or highly correlated columns
    drop_cols = [
        'Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge',
        'Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes',
        'Total day calls', 'Total eve calls', 'Total night calls', 'Total intl calls',
        'Voice mail plan', 'Weighted service calls', 'Service call share'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df


# Helper Functions

# load data in a dataframe
@st.cache_data
def load_data(csv_path="Telecom_Churn.csv"):
    return pd.read_csv(csv_path)

# describe schema of the dataframe for overview
def describe_schema(df: pd.DataFrame):
    schema = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str),
            "n_unique": [df[c].nunique() for c in df.columns],
            "missing": df.isna().sum(),
        }
    )
    return schema

# plot numeric distributions for overview
def plot_numeric_distributions(df: pd.DataFrame, numeric_columns: list):
    charts = []
    for col in numeric_columns:
        chart = (
            alt.Chart(df[[col]].dropna())
            .mark_area(opacity=0.3)
            .encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=60)),
                y=alt.Y("count()"),
            )
            .properties(height=150)
        )
        charts.append((col, chart))
    return charts

# get parameter grid for model training
def get_param_grid():
    return [
    {
        # Logistic Regression
        'classifier': [LogisticRegression(class_weight='balanced', solver='liblinear')],
        'classifier__C': [0.1, 1],
        'classifier__penalty': ['l1', 'l2']
    },
    {
        # Random Forest
        'classifier': [RandomForestClassifier(class_weight='balanced', n_jobs=-1)],
        'classifier__n_estimators': [100],
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1, 2]
    },
    {
        # SVC (RBF kernel)
        'classifier': [SVC(probability=True, class_weight='balanced')],
        'classifier__C': [0.1, 1],
        'classifier__kernel': ['rbf'],
        'classifier__gamma': ['scale']
    },
    {
        # XGBoost
        'classifier': [XGBClassifier(use_label_encoder=False, eval_metric='logloss')],
        'classifier__n_estimators': [100],
        'classifier__max_depth': [3, 7],
        'classifier__learning_rate': [0.1],
        'classifier__gamma': [0, 0.1],
        'classifier__scale_pos_weight': [6]
    }
    ]

# Train the best model using GridSearchCV for model retraining
def train_best_model(df: pd.DataFrame, target: str = "Churn"):
    pipe = Pipeline(steps=[("classifier", LogisticRegression())])  # Renamed from 'model' to 'classifier'

    param_grid = get_param_grid()
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="roc_auc",
        verbose=0,
    )
    grid.fit(X_train, y_train)
    # get the best model
    best_model = grid.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)
    return best_model, metrics, grid.best_params_


# Evaluate the model using various metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
    }
    return metrics

# Profit Analysis to evelauate the impact of churn prediction
def profit_analysis_advanced(
    df: pd.DataFrame,
    model,
    clv: float,
    discount: float,
    discount_success_prob: float,
    threshold: float = 0.5,
    target: str = "Churn",
):
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    proba = model.predict_proba(X)[:, 1]
    send_discount = proba >= threshold

    profit_blanket = 0
    for churn in y:
        if churn == 1:
            profit_blanket += (clv * discount_success_prob) - discount
        else:
            profit_blanket += clv - discount

    profit_model = 0
    for actual, predicted in zip(y, send_discount):
        if predicted:
            if actual == 1:
                profit_model += (clv * discount_success_prob) - discount
            else:
                profit_model += clv - discount
        else:
            if actual == 0:
                profit_model += clv
            # If actual == 1 and no discount → no profit

    return profit_blanket, profit_model, profit_model - profit_blanket

# User Input for Prediction
def get_user_input():
    with st.form("prediction_form"):
        st.subheader("Provide customer information")

        input_data = {
            "Account length": st.number_input("Account length", min_value=1, value=100),
            "International plan": st.selectbox("International plan", ["Yes", "No"]),
            "Voice mail plan": st.selectbox("Voice mail plan", ["Yes", "No"]),
            "Number vmail messages": st.number_input("Number vmail messages", min_value=0, value=0),
            "Total day minutes": st.number_input("Total day minutes", min_value=0.0, value=120.0),
            "Total day calls": st.number_input("Total day calls", min_value=0, value=100),
            "Total day charge": st.number_input("Total day charge", min_value=0.0, value=30.0),
            "Total eve minutes": st.number_input("Total eve minutes", min_value=0.0, value=120.0),
            "Total eve calls": st.number_input("Total eve calls", min_value=0, value=100),
            "Total eve charge": st.number_input("Total eve charge", min_value=0.0, value=20.0),
            "Total night minutes": st.number_input("Total night minutes", min_value=0.0, value=120.0),
            "Total night calls": st.number_input("Total night calls", min_value=0, value=100),
            "Total night charge": st.number_input("Total night charge", min_value=0.0, value=10.0),
            "Total intl minutes": st.number_input("Total intl minutes", min_value=0.0, value=10.0),
            "Total intl calls": st.number_input("Total intl calls", min_value=0, value=5),
            "Total intl charge": st.number_input("Total intl charge", min_value=0.0, value=3.0),
            "Customer service calls": st.number_input("Customer service calls", min_value=0, value=1)
        }

        submitted = st.form_submit_button("Predict")
        if submitted:
            return pd.DataFrame([input_data])
    return None

# plot confision matrix and ROC curve
def draw_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax)
    st.pyplot(fig)

def draw_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    st.pyplot(fig)


# Update save_model to default to new path
def save_model(model, path="churn_prediction_model.pkl"):
    joblib.dump(model, path)

# Update load_model to match new filename
def load_model(path="churn_prediction_model.pkl"):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# Main App

# Load data and model
raw_data = load_data()
data = transform_features(raw_data)

numeric_cols = data.select_dtypes(include=["int64", "float64"]).drop(columns=["Churn"], errors="ignore").columns.tolist()
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

model = load_model()

# initialize model if not already loaded
if model is None:
    with st.spinner("Training initial model..."):
        model, metrics, _ = train_best_model(data)
        save_model(model)

# setting up sidebar and page navigaiontion
st.sidebar.title("Navigation")
selection = st.sidebar.radio(
    "Go to", ["Data Overview", "Predict", "Model Evaluation", "Retrain Model"]
)

# Sidebar styling
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
        }
        section[data-testid="stSidebar"] .stRadio label {
            font-weight: 500;
            color: inherit;
        }
        section[data-testid="stSidebar"] p {
            font-size: 0.85rem;
            color: var(--text-color);
            opacity: 0.8;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--text-color);
        }
        .main .block-container {
            padding-top: 1rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Data Overview page
if selection == "Data Overview":
    st.title("Dataset Overview")
    schema = describe_schema(data)
    st.subheader("Data Schema")
    st.dataframe(schema)

    # illustratiing nuerical feature distributions
    st.subheader("Numerical Feature Distributions")
    charts = plot_numeric_distributions(data, numeric_cols)
    cols = st.columns(2)
    for idx, (name, chart) in enumerate(charts):
        with cols[idx % 2]:
            st.altair_chart(chart, use_container_width=True)

# Prediction page
elif selection == "Predict":
    st.title("Predict Customer Churn")

    # two tabs: for single and batch prediction
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])
    # tab for single prediction
    with tab1:
        input_df = get_user_input()
        if input_df is not None:
            transformed_input = transform_features(input_df)
            pred_proba = model.predict_proba(transformed_input)[0, 1]
            pred_label = "Churn" if pred_proba >= 0.5 else "Stay"
            st.success(f"Prediction: **{pred_label}** (probability {pred_proba:.2%})")

            # Save to session history
            st.session_state.prediction_history.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input": input_df.to_dict(orient="records")[0],
                "prediction": pred_label,
                "probability": round(pred_proba, 4)
            })
        # display prediction history
        if st.session_state.prediction_history:
            with st.expander("View Prediction History", expanded=False):
                history_df = pd.DataFrame([
                    {
                        **entry["input"],
                        "Prediction": entry["prediction"],
                        "Probability": entry["probability"],
                        "Timestamp": entry["timestamp"]
                    }
                    for entry in st.session_state.prediction_history
                ])
                st.dataframe(history_df)

    # second tab for batch prediction using csv file upload
    with tab2:
        st.subheader("Upload CSV for Batch Predictions")

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)

                # Validate required columns
                required_cols = [
                    "Account length", "International plan", "Voice mail plan",
                    "Number vmail messages", "Total day minutes", "Total day calls", "Total day charge",
                    "Total eve minutes", "Total eve calls", "Total eve charge",
                    "Total night minutes", "Total night calls", "Total night charge",
                    "Total intl minutes", "Total intl calls", "Total intl charge",
                    "Customer service calls"
                ]

                if not all(col in df_upload.columns for col in required_cols):
                    st.error("Uploaded file is missing required columns.")
                else:
                    # Transform and predict
                    df_transformed = transform_features(df_upload)
                    probs = model.predict_proba(df_transformed)[:, 1]
                    preds = ["Churn" if p >= 0.5 else "Stay" for p in probs]

                    df_upload["Prediction"] = preds
                    df_upload["Probability"] = probs

                    st.success(f"Predictions complete for {len(df_upload)} rows.")
                    st.dataframe(df_upload)

                    # Optional download
                    csv_out = df_upload.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results as CSV",
                        data=csv_out,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")

# Model Evaluation page
elif selection == "Model Evaluation":
    st.title("Model Evaluation")
    metrics = evaluate_model(
        model, data.drop(columns=["Churn"]), data["Churn"].astype(int)
    )
    # Display metrics in table
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    st.metric("AUC", f"{metrics['auc']:.2f}")

    # show confusion matrix and ROC curve
    st.subheader("Confusion Matrix")
    draw_confusion_matrix(metrics["cm"])

    st.subheader("ROC Curve")
    draw_roc_curve(metrics["fpr"], metrics["tpr"], metrics["auc"])

    st.subheader("Profit Analysis")

    # User input for profit analysis allowing to evaluate the impact of churn prediction
    with st.expander("Assumptions"):
        st.markdown("""
        **Two strategies are compared to evaluate the impact of churn prediction:**

        - **Blanket Strategy**:  
        Every customer receives a discount.  
        - If they are a *churner*, the discount has a certain probability of retaining them.
        - If they are *not a churner*, the discount is unnecessary but still applied.

        - **Model-Based Strategy**:  
        Only customers **predicted to churn** receive the discount.  
        - If the prediction is correct, it may retain the customer.
        - If it's a false alarm, the discount is wasted.
        - If a churner is missed, no profit is recovered.

        In both strategies, we compute the **expected profit** using the assumed retention probability.
        """)
        monthly_revenue = st.number_input("Monthly recurring revenue (€)", value=30.0)
        retention_months = st.slider("Expected months retained if saved", 1, 36, 12)
        discount = st.number_input("Discount amount per customer (€)", value=10.0)
        success_prob = st.slider("Probability discount retains churner (%)", 0.0, 100.0, 30.0) / 100
        threshold = st.slider("Churn probability threshold", 0.0, 1.0, 0.5, 0.01)

    expected_clv = monthly_revenue * retention_months

    # Calculate profits
    blanket, model_based, delta = profit_analysis_advanced(
        data, model,
        clv=expected_clv,
        discount=discount,
        discount_success_prob=success_prob,
        threshold=threshold
    )
    # Display results in table
    col1, col2, col3 = st.columns(3)
    col1.metric("Blanket Strategy", f"€{blanket:,.0f}")
    col2.metric("Model Strategy", f"€{model_based:,.0f}")
    col3.metric("Δ Profit", f"€{delta:,.0f}")

# retain model page
elif selection == "Retrain Model":
    st.title("Retrain Model")
    st.markdown("""
        You can retrain the model using the latest available dataset and 
        compare results across training sessions.
    """)
    st.markdown("---")

    st.header("Current Training Run")
    st.write("This will run a grid search to find the best hyperparameters.")

    if st.button("Start Retraining"):
        with st.spinner("Retraining in progress..."):
            new_model, new_metrics, best_params = train_best_model(data)
            save_model(new_model)  
            model = new_model

            # Store retraining metadata in history
            st.session_state.retraining_history.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_type": type(new_model.named_steps['classifier']).__name__,
                "params": best_params,
                "metrics": {
                    "accuracy": f"{new_metrics['accuracy']:.2%}",
                    "precision": f"{new_metrics['precision']:.2%}",
                    "recall": f"{new_metrics['recall']:.2%}",
                    "auc": f"{new_metrics['auc']:.2%}"
                }
            })

            # UI Feedback
            st.success("Model retrained successfully!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{new_metrics['accuracy']:.2%}")
            col2.metric("Precision", f"{new_metrics['precision']:.2%}")
            col3.metric("Recall", f"{new_metrics['recall']:.2%}")
            col4.metric("AUC", f"{new_metrics['auc']:.2f}")

            st.markdown("### 🔧 Best Hyperparameters")
            for k, v in best_params.items():
                st.markdown(f"- **{k}**: `{v}`")

    # Retraining history display
    if "retraining_history" in st.session_state and st.session_state.retraining_history:
        st.markdown("---")
        with st.expander("View Retraining History", expanded=False):
            for record in reversed(st.session_state.retraining_history):
                st.markdown(f"**Timestamp:** {record['timestamp']}")
                st.markdown(f"**Model Type:** {record['model_type']}")
                st.markdown("**Best Parameters:**")
                st.json(record["params"])
                st.markdown("**Metrics:**")
                st.json(record["metrics"])
                st.markdown("---")
    else:
        st.info("No retraining history yet.")


