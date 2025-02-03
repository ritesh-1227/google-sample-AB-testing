# src/hypothesis_recommendation.py

from typing import Optional

import numpy as np
import pandas as pd

from src.ab_testing import ABTest


def assign_recommendation_groups(
    user_df: pd.DataFrame, user_id_col: str = "fullVisitorId", seed: int = 42
) -> pd.DataFrame:
    """
    Randomly assigns half the users to 'control' (generic recs) and half to 'test' (personalized recs).
    Appends a new column 'rec_group' with values 'control' or 'test'.
    """
    df = user_df.copy()
    unique_users = df[user_id_col].unique()
    rng = np.random.default_rng(seed)

    mask = rng.random(len(unique_users)) < 0.5
    control_users = unique_users[mask]
    test_users = unique_users[~mask]

    # Create a map user -> group
    group_map = {}
    for uid in control_users:
        group_map[uid] = "control"
    for uid in test_users:
        group_map[uid] = "test"

    df["rec_group"] = df[user_id_col].map(group_map)
    return df


def run_recommendation_test(
    user_df: pd.DataFrame,
    user_id_col: str = "fullVisitorId",
    metric_col: str = "totalTransactionRevenue",
    test_type: str = "t_test",
    transform: str = "none",
    zero_inflation: bool = False,
) -> dict:
    """
    Conduct an A/B test on the 'metric_col' to see if personalized recs (test)
    outperform generic recs (control) after random assignment.
    """
    assigned_df = assign_recommendation_groups(user_df, user_id_col=user_id_col)

    # Build ABTest
    ab = ABTest(
        df=assigned_df,
        control_filter={"rec_group": "control"},
        test_filter={"rec_group": "test"},
    )

    result = ab.run_test(
        column=metric_col,
        test_type=test_type,
        transform=transform,
        zero_inflation=zero_inflation,
    )
    return result
