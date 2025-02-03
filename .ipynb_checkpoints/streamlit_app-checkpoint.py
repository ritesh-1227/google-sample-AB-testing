import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ab_test_reporting import interpret_ab_results
from src.hypothesis_pricing import run_pricing_test
# Hypothesis modules & reporting
from src.hypothesis_recommendation import run_recommendation_test

###############################################################################
#  Utility Functions for Interactive Plots (Funnel, Campaign, Country, etc.)
###############################################################################


def compute_funnel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example function that computes a simple funnel:
      1) Total unique users
      2) Users with transactions > 0
      3) Total sum of transactions
    """
    if "fullVisitorId" in df.columns:
        total_users = df["fullVisitorId"].nunique()
    else:
        total_users = len(df)

    if "transactions" in df.columns:
        users_with_txn = df[df["transactions"] > 0]["fullVisitorId"].nunique()
        total_txn = df["transactions"].sum()
    else:
        users_with_txn = 0
        total_txn = 0

    funnel_data = pd.DataFrame(
        {
            "Stage": ["All Users", "Users w/ Transactions", "Total Txn Count"],
            "Value": [total_users, users_with_txn, total_txn],
        }
    )
    return funnel_data


def plot_funnel(funnel_data: pd.DataFrame):
    """Use Plotly to create a funnel chart from funnel_data."""
    fig = px.funnel(funnel_data, y="Stage", x="Value", title="Conversion Funnel")
    return fig


def campaign_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total transactions by trafficSource.
    """
    if "trafficSource" not in df.columns or "transactions" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("trafficSource", as_index=False).agg({"transactions": "sum"})
    grouped.rename(columns={"transactions": "TotalTransactions"}, inplace=True)
    return grouped


def plot_campaign_bar(campaign_df: pd.DataFrame):
    """
    Creates a bar plot of total transactions by trafficSource.
    """
    fig = px.bar(
        campaign_df,
        x="trafficSource",
        y="TotalTransactions",
        title="Campaign Effectiveness",
    )
    fig.update_layout(xaxis_title="Traffic Source", yaxis_title="Transactions")
    return fig


def country_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sum of transactions by country.
    """
    if "country" not in df.columns or "transactions" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("country", as_index=False).agg({"transactions": "sum"})
    grouped.rename(columns={"transactions": "TotalTransactions"}, inplace=True)
    return grouped


def plot_country_bar(country_df: pd.DataFrame):
    """
    Creates a bar chart showing total transactions by country, but only for top 5.
    Shows a bar chart of top 5, but the entire DataFrame can be displayed in a table.
    """
    # Get top 5 countries by TotalTransactions
    top_5 = country_df.nlargest(5, "TotalTransactions")

    fig = px.bar(
        top_5,
        x="country",
        y="TotalTransactions",
        title="Top 5 Countries by Transactions",
    )
    fig.update_layout(xaxis_title="Country", yaxis_title="Transactions")
    return fig


###############################################################################
#                                Streamlit App
###############################################################################


def main():
    st.title("Interactive Analytics & A/B Testing App")

    # Sidebar data source
    st.sidebar.header("Data Source")
    data_choice = st.sidebar.selectbox(
        "Select Data Source", ["Local CSV", "Upload CSV"]
    )

    # Step 1: Load Data
    if data_choice == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your user-level CSV")
        if not uploaded_file:
            st.warning("Please upload a CSV to proceed.")
            st.stop()
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV loaded successfully.")
    else:
        local_path = "data/user_agg_cleaned.csv"
        st.sidebar.write(f"Using local CSV: `{local_path}`")
        if not os.path.exists(local_path):
            st.error(
                f"No local file found at {local_path}. Please place user_level.csv in data/ folder."
            )
            st.stop()
        df = pd.read_csv(local_path)

    st.write("## Data Preview")
    st.write(df.head(10))
    st.write(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Show summary stats
    if st.checkbox("Show summary stats"):
        st.write(df.describe(include="all"))

    # Basic EDA - histogram
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.sidebar.subheader("Histogram")
        hist_col = st.sidebar.selectbox("Numeric column for histogram", numeric_cols)
        if st.sidebar.button("Plot Histogram"):
            fig = px.histogram(
                df, x=hist_col, nbins=30, title=f"Distribution of {hist_col}"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Conversion Funnel")
    funnel_data = compute_funnel_data(df)
    if st.button("Show Funnel"):
        funnel_fig = plot_funnel(funnel_data)
        st.plotly_chart(funnel_fig, use_container_width=True)
        st.write(funnel_data)

    st.markdown("---")
    st.header("Campaign Effectiveness")
    # If trafficSource exists, we show bar chart
    if "trafficSource" in df.columns and "transactions" in df.columns:
        campaign_df = campaign_effectiveness(df)
        if not campaign_df.empty:
            if st.button("Show Campaign Chart"):
                camp_fig = plot_campaign_bar(campaign_df)
                st.plotly_chart(camp_fig, use_container_width=True)
                st.write(campaign_df)
        else:
            st.write(
                "No valid trafficSource or transactions columns to show campaign data."
            )
    else:
        st.write(
            "No 'trafficSource' or 'transactions' columns found for campaign analysis."
        )

    st.markdown("---")
    st.header("Country-wise Conversion")
    if "country" in df.columns and "transactions" in df.columns:
        country_df = country_conversion(df)
        if not country_df.empty:
            if st.button("Show Country Chart"):
                # Plot top 5
                country_fig = plot_country_bar(country_df)
                st.plotly_chart(country_fig, use_container_width=True)

                # Display entire country data below the chart
                st.write("#### All Countries Data")
                st.dataframe(country_df)
        else:
            st.write(
                "No valid 'country' or 'transactions' columns to show country data."
            )
    else:
        st.write(
            "No 'country' or 'transactions' columns found for country-wise analysis."
        )

    st.markdown("---")
    st.header("Hypothesis & A/B Testing")

    # Let user pick which hypothesis to test
    hypothesis_choice = st.selectbox(
        "Select a Hypothesis", ["Product Recommendation", "Dynamic Pricing"]
    )

    # Common test parameters
    test_type = st.selectbox(
        "Test Type", ["t_test", "mannwhitney", "bayesian_conversions", "bayesian_means"]
    )
    transform = st.selectbox(
        "Transformation", ["none", "log", "winsor", "trim", "boxcox"], index=0
    )
    zero_inflation = st.checkbox("Zero Inflation?", value=False)

    # Product Recommendation Hypothesis
    if hypothesis_choice == "Product Recommendation":
        st.info(
            "Randomly assign half users to 'control' (generic) & half to 'test' (personalized)."
        )
        metric_col = st.selectbox(
            "Metric to Compare",
            numeric_cols,
            index=numeric_cols.index("totalTransactionRevenue")
            if "totalTransactionRevenue" in numeric_cols
            else 0,
        )

        if st.button("Run Product Recommendation Test"):
            result = run_recommendation_test(
                user_df=df,
                user_id_col="fullVisitorId",
                metric_col=metric_col,
                test_type=test_type,
                transform=transform,
                zero_inflation=zero_inflation,
            )
            st.subheader("Test Results")
            st.text(interpret_ab_results(result))

    # Dynamic Pricing Hypothesis
    elif hypothesis_choice == "Dynamic Pricing":
        st.info(
            "Assign users to 'test' if totalTransactionRevenue >= threshold => dynamic pricing, else 'control'."
        )
        threshold_val = st.number_input(
            "Revenue Threshold for test group", value=200.0, step=50.0
        )
        metric_col = st.selectbox(
            "Metric to Compare",
            numeric_cols,
            index=numeric_cols.index("transactions")
            if "transactions" in numeric_cols
            else 0,
        )

        if st.button("Run Dynamic Pricing Test"):
            result = run_pricing_test(
                user_df=df,
                threshold=threshold_val,
                metric_col=metric_col,
                test_type=test_type,
                transform=transform,
                zero_inflation=zero_inflation,
            )
            st.subheader("Test Results")
            st.text(interpret_ab_results(result))

        # ...
    st.markdown("---")
    st.markdown(
        """
    **Notes & Future Work**

    - **Load user-level data** (upload or local).
      <span style="color:#FF4B4B; font-weight:bold;">Note:</span> Currently, this is only used for a specific learning experience.
      It is not automated for all datasets and works for only specific data I used.

    - **Perform basic EDA** with histograms, funnel, campaign, and country analysis.

    - **Run hypothesis-driven A/B tests** (Product Recommendation, Dynamic Pricing).

    - **Present results** in a user-friendly format with optional transformations & zero inflation handling.

    - You can expand these sections or add new hypotheses & analyses as needed.
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
