from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Academic Performance Analysis",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Student Academic Performance Analysis & Prediction")
st.caption(
    "Explore student data, understand important factors, compare regression "
    "models, and estimate an exam score using the project's dataset."
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "dataset" / "student_performance.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Make sure "
            "'dataset/student_performance.csv' exists in the GitHub repository."
        )

    data = pd.read_csv(data_path)

    if "Exam_Score" not in data.columns:
        raise ValueError("The dataset must contain an 'Exam_Score' column.")

    data = data.drop_duplicates().copy()
    data["Exam_Score"] = pd.to_numeric(data["Exam_Score"], errors="coerce")
    data = data.dropna(subset=["Exam_Score"])

    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the project dataset: {error}")
    st.stop()


target = "Exam_Score"
X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()


# ---------------------------------------------------------
# Build preprocessing and regression models
# ---------------------------------------------------------
def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])


@st.cache_resource
def train_models(dataframe):
    features = dataframe.drop(columns=[target])
    labels = dataframe[target]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42
    )

    model_configs = {
        "Multiple Linear Regression": LinearRegression(),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    trained = {}
    metric_rows = {}

    for model_name, estimator in model_configs.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        trained[model_name] = pipeline
        metric_rows[model_name] = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R² Score": r2_score(y_test, predictions)
        }

    metrics = pd.DataFrame(metric_rows).T
    return trained, metrics


with st.spinner("Preparing the machine learning models..."):
    models, model_metrics = train_models(df)


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Dataset Explorer",
        "Visualizations",
        "Model Performance",
        "Exam Score Prediction"
    ]
)


# ---------------------------------------------------------
# Project overview
# ---------------------------------------------------------
if page == "Project Overview":
    st.header("Project Overview")
    st.write(
        "This project studies student academic performance using data analysis "
        "and machine learning. The target variable is **Exam_Score**."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Features", f"{X.shape[1]}")
    col3.metric("Target Variable", target)

    st.subheader("Project Workflow")
    st.markdown(
        """
        1. Dataset inspection and data quality assessment
        2. Data cleaning and preprocessing
        3. Exploratory data analysis and visualization
        4. Regression model training and evaluation
        5. Student exam score estimation
        """
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df[target], kde=True, ax=ax)
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------
# Dataset explorer
# ---------------------------------------------------------
elif page == "Dataset Explorer":
    st.header("Dataset Explorer")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]}")
    col3.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.subheader("Filter Dataset")
    selected_columns = st.multiselect(
        "Choose columns to display",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )

    if selected_columns:
        st.dataframe(df[selected_columns], use_container_width=True)
    else:
        st.info("Select at least one column to display.")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values remain in the displayed dataset.")
    else:
        st.dataframe(
            missing.rename("Missing Values").to_frame(),
            use_container_width=True
        )


# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------
elif page == "Visualizations":
    st.header("Explore the Data")

    st.subheader("Exam Score Distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df[target], kde=True, ax=ax)
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)

    if numeric_features:
        selected_numeric = st.selectbox(
            "Select a numerical feature",
            numeric_features
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.scatterplot(data=df, x=selected_numeric, y=target, alpha=0.45, ax=ax)
        ax.set_title(f"{selected_numeric} vs Exam Score")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation Heatmap")
    numeric_data = df.select_dtypes(include=np.number)

    if numeric_data.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(11, 7))
        sns.heatmap(
            numeric_data.corr(),
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            ax=ax
        )
        st.pyplot(fig)
        plt.close(fig)

    if categorical_features:
        st.subheader("Categorical Feature Distribution")
        selected_category = st.selectbox(
            "Select a categorical feature",
            categorical_features
        )

        category_counts = (
            df[selected_category]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            x=category_counts.values,
            y=category_counts.index,
            ax=ax
        )
        ax.set_xlabel("Number of Students")
        ax.set_ylabel(selected_category)
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------
elif page == "Model Performance":
    st.header("Regression Model Performance")
    st.write(
        "Models are evaluated on a held-out test set (20%) using the same "
        "random state as the notebook's train-test split."
    )

    st.dataframe(
        model_metrics.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "R² Score": "{:.3f}"
        }),
        use_container_width=True
    )

    best_model = model_metrics["R² Score"].idxmax()
    st.info(f"Highest test-set R² score in this app: **{best_model}**")

    st.caption(
        "MAE and RMSE are error measures (lower is better). R² measures the "
        "proportion of target variance explained by the model (higher is better)."
    )


# ---------------------------------------------------------
# Interactive prediction
# ---------------------------------------------------------
elif page == "Exam Score Prediction":
    st.header("Estimate a Student's Exam Score")
    st.write(
        "Enter student details below. The app uses the selected regression "
        "model, trained on the project's dataset, to estimate an exam score."
    )

    selected_model_name = st.selectbox(
        "Choose a regression model",
        list(models.keys())
    )

    input_values = {}

    with st.form("student_prediction_form"):
        st.subheader("Student Information")
        columns = st.columns(2)

        for index, feature in enumerate(X.columns):
            container = columns[index % 2]
            series = X[feature]

            if feature in numeric_features:
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()

                if numeric_series.empty:
                    input_values[feature] = container.number_input(
                        feature, value=0.0
                    )
                else:
                    minimum = float(numeric_series.min())
                    maximum = float(numeric_series.max())
                    median = float(numeric_series.median())

                    if pd.api.types.is_integer_dtype(series.dtype):
                        input_values[feature] = container.number_input(
                            feature,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(round(median)),
                            step=1
                        )
                    else:
                        input_values[feature] = container.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=float(median),
                            step=0.5
                        )
            else:
                options = sorted(
                    series.dropna().astype(str).unique().tolist()
                )

                if not options:
                    options = ["Unknown"]

                input_values[feature] = container.selectbox(
                    feature,
                    options
                )

        submitted = st.form_submit_button(
            "Predict Exam Score",
            type="primary",
            use_container_width=True
        )

    if submitted:
        input_df = pd.DataFrame([input_values], columns=X.columns)
        prediction = models[selected_model_name].predict(input_df)[0]

        st.success("Prediction completed.")
        st.metric(
            "Estimated Exam Score",
            f"{prediction:.2f}"
        )

        st.caption(
            "This is a model-based estimate, not a guaranteed or official exam result."
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "Student Academic Performance Analysis & Prediction | "
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)

