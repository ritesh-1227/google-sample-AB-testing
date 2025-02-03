# src/hypothesis_pricing.py

from typing import Optional

import numpy as np
import pandas as pd

from src.ab_testing import ABTest


def assign_pricing_groups(
    user_df: pd.DataFrame, threshold: float = 200.0, user_id_col: str = "fullVisitorId"
) -> pd.DataFrame:
    """
    Assigns users to 'test' if totalTransactionRevenue >= threshold, else 'control'.
    This is a contrived approach to demonstrate code, not a true random assignment.
    """
    df = user_df.copy()
    if "totalTransactionRevenue" not in df.columns:
        raise ValueError(
            "DataFrame missing 'totalTransactionRevenue' for pricing logic"
        )

    df["price_group"] = np.where(
        df["totalTransactionRevenue"] >= threshold, "test", "control"
    )
    return df


def run_pricing_test(
    user_df: pd.DataFrame,
    threshold: float = 200.0,
    metric_col: str = "transactions",
    test_type: str = "mannwhitney",
    transform: str = "none",
    zero_inflation: bool = True,
    user_id_col: str = "fullVisitorId",
) -> dict:
    """
    Test whether dynamic pricing (assigned to 'test' if totalRevenue >= threshold)
    leads to higher 'metric_col' than fixed pricing.
    By default, uses Mann-Whitney if data is skewed.
    """
    assigned_df = assign_pricing_groups(
        user_df, threshold=threshold, user_id_col=user_id_col
    )

    ab = ABTest(
        df=assigned_df,
        control_filter={"price_group": "control"},
        test_filter={"price_group": "test"},
    )

    result = ab.run_test(
        column=metric_col,
        test_type=test_type,
        transform=transform,
        zero_inflation=zero_inflation,
    )
    return result
