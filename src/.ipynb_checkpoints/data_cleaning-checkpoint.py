# src/data_cleaning.py

import numpy as np
import pandas as pd


def profile_data(df: pd.DataFrame) -> dict:
    """
    Profiles the DataFrame and returns a summary dictionary including:
    - Missing values per column
    - Descriptive statistics for numeric columns
    - Outlier detection (using the IQR method) for each numeric column
    """
    profile = {}

    # Missing values per column
    profile["missing_values"] = df.isnull().sum().to_dict()

    # Descriptive statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[float, int, np.number]).columns
    profile["descriptive_stats"] = df[numeric_cols].describe().to_dict()

    # Outlier detection using IQR for each numeric column
    outlier_summary = {}
    for col in numeric_cols:
        valid_vals = df[col].dropna()
        if len(valid_vals) == 0:
            outlier_summary[col] = {
                "count": 0,
                "min": None,
                "max": None,
                "sample_values": [],
            }
            continue

        Q1 = valid_vals.quantile(0.25)
        Q3 = valid_vals.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = valid_vals[(valid_vals < lower_bound) | (valid_vals > upper_bound)]
        outlier_summary[col] = {
            "count": int(outliers.count()),
            "min": float(outliers.min()) if not outliers.empty else None,
            "max": float(outliers.max()) if not outliers.empty else None,
            "sample_values": outliers.head(5).tolist(),
        }
    profile["outliers"] = outlier_summary

    return profile


def clean_sessions_data(
    df: pd.DataFrame, cleaning_options: dict = None
) -> pd.DataFrame:
    """
    Clean and format the sessions DataFrame with the following categorical mappings:
      1) city:          ("not available in demo dataset", "(not set)") => "NotSet"
      2) country:       (none, (not set), "not available in demo dataset") => "NotSet"
      3) trafficCampaign:   (not set) => "NotSet"
      4) trafficMedium:     ((none), (not set)) => "NotSet"
      5) trafficSource:     Keep the top 5 categories by frequency; rest => "Others"

    Numeric columns that might be <NA> or Int64 will be converted to float columns with NaN.

    Optional cleaning steps (all default to False):
      - 'convert_date': bool. Convert 'date' to datetime.
      - 'revenue_adjustment': bool. Convert 'totalTransactionRevenue' from micros if large.
      - 'fill_missing_transactions': bool. Fill NaN in 'transactions' with 0.
      - 'fill_missing_pageviews': bool. Fill NaN in 'pageviews' with 0.
      - 'fill_missing_timeOnSite': bool. Fill NaN in 'timeOnSite' with 0.
      - 'timeOnSite_zero_floor': bool. If True, negative timeOnSite => 0.
    """
    default_opts = {
        "convert_date": False,
        "revenue_adjustment": False,
        "fill_missing_transactions": False,
        "fill_missing_pageviews": False,
        "fill_missing_timeOnSite": False,
        "timeOnSite_zero_floor": False,
    }
    if cleaning_options is None:
        cleaning_options = {}
    options = {**default_opts, **cleaning_options}

    # 1) Convert possible Int64 <NA> columns to float with NaN.
    #    This ensures we use standard NaN for missing numeric data.
    numeric_cols = [
        "pageviews",
        "timeOnSite",
        "transactions",
        "totalTransactionRevenue",
    ]
    for col in numeric_cols:
        if col in df.columns and pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)  # <NA> => np.nan

    # 2) Map specific string values to "NotSet" for city, country, trafficCampaign, trafficMedium.
    #    We'll unify them in a single code block, but each column has its own rules.

    # city -> "NotSet" if "not available in demo dataset" or "(not set)"
    if "city" in df.columns:
        df["city"] = df["city"].fillna("NotSet")  # in case it's truly NaN
        df.loc[
            df["city"].isin(["not available in demo dataset", "(not set)"]), "city"
        ] = "NotSet"

    # country -> "NotSet" if "not available in demo dataset" or "(not set)" or "(none)"
    if "country" in df.columns:
        df["country"] = df["country"].fillna("NotSet")
        df.loc[
            df["country"].isin(
                ["not available in demo dataset", "(not set)", "(none)"]
            ),
            "country",
        ] = "NotSet"

    # trafficCampaign -> "NotSet" if "(not set)"
    if "trafficCampaign" in df.columns:
        df["trafficCampaign"] = df["trafficCampaign"].fillna("NotSet")
        df.loc[df["trafficCampaign"].isin(["(not set)"]), "trafficCampaign"] = "NotSet"

    # trafficMedium -> "NotSet" if "(none)" or "(not set)"
    if "trafficMedium" in df.columns:
        df["trafficMedium"] = df["trafficMedium"].fillna("NotSet")
        df.loc[
            df["trafficMedium"].isin(["(none)", "(not set)"]), "trafficMedium"
        ] = "NotSet"

    # 3) trafficSource -> Keep the top 5 categories, rest => "Others"
    if "trafficSource" in df.columns:
        df["trafficSource"] = df["trafficSource"].fillna("NotSet")
        top_5_sources = df["trafficSource"].value_counts().nlargest(5).index
        df.loc[~df["trafficSource"].isin(top_5_sources), "trafficSource"] = "Others"

    # 4) Optional steps

    # 4.1) Convert 'date' column to datetime
    if options["convert_date"] and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # 4.2) Adjust revenue if likely in micros
    if options["revenue_adjustment"] and "totalTransactionRevenue" in df.columns:
        max_revenue = df["totalTransactionRevenue"].dropna().max()
        if max_revenue and max_revenue > 100000:
            df["totalTransactionRevenue"] = df["totalTransactionRevenue"] / 1e6
        df["totalTransactionRevenue"] = df["totalTransactionRevenue"].fillna(0)

    # 4.3) Fill missing values in numeric columns if requested
    if options["fill_missing_transactions"] and "transactions" in df.columns:
        df["transactions"] = df["transactions"].fillna(0)

    if options["fill_missing_pageviews"] and "pageviews" in df.columns:
        df["pageviews"] = df["pageviews"].fillna(0)

    if options["fill_missing_timeOnSite"] and "timeOnSite" in df.columns:
        df["timeOnSite"] = df["timeOnSite"].fillna(0)

    # 4.4) Floor negative timeOnSite to 0
    if options["timeOnSite_zero_floor"] and "timeOnSite" in df.columns:
        negative_mask = df["timeOnSite"] < 0
        df.loc[negative_mask, "timeOnSite"] = 0

    df.reset_index(drop=True, inplace=True)
    return df
