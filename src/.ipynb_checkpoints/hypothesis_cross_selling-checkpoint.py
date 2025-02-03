# src/hypothesis_cross_selling.py

from typing import Optional

import numpy as np
import pandas as pd

from src.ab_testing import ABTest


def assign_cross_sell_groups(
    user_df: pd.DataFrame,
    item_x_col: str = "bought_item_x",
    user_id_col: str = "fullVisitorId",
) -> pd.DataFrame:
    """
    If user bought item X => 'test' (cross-sell shown), else 'control'.
    This is a simplistic approach, not random but purely condition-based.
    """
    df = user_df.copy()
    if item_x_col not in df.columns:
        raise ValueError(
            f"DataFrame missing '{item_x_col}' column for cross-selling logic"
        )

    df["cross_sell_group"] = np.where(df[item_x_col] == 1, "test", "control")
    return df


def run_cross_sell_test(
    user_df: pd.DataFrame,
    item_x_col: str = "bought_item_x",
    metric_col: str = "totalTransactionRevenue",
    test_type: str = "t_test",
    transform: str = "none",
    zero_inflation: bool = True,
    user_id_col: str = "fullVisitorId",
) -> dict:
    """
    Test cross-selling hypothesis: users who bought item X => show cross-sell => 'test',
    compare 'metric_col' to 'control' group (not shown cross-sell).
    """
    assigned_df = assign_cross_sell_groups(user_df, item_x_col, user_id_col)

    ab = ABTest(
        df=assigned_df,
        control_filter={"cross_sell_group": "control"},
        test_filter={"cross_sell_group": "test"},
    )

    result = ab.run_test(
        column=metric_col,
        test_type=test_type,
        transform=transform,
        zero_inflation=zero_inflation,
    )
    return result
