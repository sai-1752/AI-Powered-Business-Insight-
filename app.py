import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from insight_engine import generate_insights

st.title("AI CSV Insight Generator Dashboard")

def generate_charts(df):

    numeric_cols = df.select_dtypes(include=['float64','int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # ==============================
    # Distribution + Outlier Insight
    # ==============================
    if len(numeric_cols) >= 1:
        st.subheader("KPI Distribution & Outlier Detection")

        fig, ax = plt.subplots(1,2, figsize=(12,4))

        sns.histplot(df[numeric_cols[0]],
                     kde=True,
                     color="#4CAF50",
                     ax=ax[0])
        ax[0].set_title("Distribution")

        sns.boxplot(x=df[numeric_cols[0]],
                    color="#FF9800",
                    ax=ax[1])
        ax[1].set_title("Outlier View")

        st.pyplot(fig)


    # ==============================
    # Top Category Performance
    # ==============================
    if len(categorical_cols) >= 1:
        st.subheader("Top Category Performance")

        top_categories = df[categorical_cols[0]].value_counts().nlargest(10)

        fig, ax = plt.subplots(figsize=(8,4))

        sns.barplot(x=top_categories.values,
                    y=top_categories.index,
                    palette="viridis",
                    ax=ax)

        ax.set_title("Top Performing Categories")
        st.pyplot(fig)


    # ==============================
    # Enhanced Correlation Heatmap
    # ==============================
    if len(numeric_cols) >= 2:
        st.subheader("Feature Correlation Matrix")

        corr = df[numeric_cols].corr()
        mask = np.triu(corr)

        fig, ax = plt.subplots(figsize=(8,6))

        sns.heatmap(corr,
                    mask=mask,
                    annot=True,
                    cmap="coolwarm",
                    linewidths=0.5,
                    fmt=".2f",
                    ax=ax)

        st.pyplot(fig)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview", df.head())
 
    generate_charts(df)

    summary = df.describe().to_string()

    if st.button("Generate AI Insights"):
        insights = generate_insights(summary)
        st.subheader("AI Generated Report")
        st.write(insights)