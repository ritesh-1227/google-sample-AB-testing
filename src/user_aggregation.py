"""
user_aggregation.py

A comprehensive module to aggregate session-level (visit-level) data to user-level rows,
while intelligently handling numeric, date/time, and categorical columns.

Author: [Your Name]
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def aggregate_user_data(
    df: pd.DataFrame,
    user_id_col: str,
    group_col: Optional[str] = None,
    handle_multi_group: str = "exclude",
    numeric_strategy: str = "sum",
    date_strategy: str = "min",
    categorical_strategy: str = "majority",
    custom_strategies: Optional[Dict[str, Callable[[pd.Series], any]]] = None,
    exclude_columns: Optional[List[str]] = None,
    datetime_formats: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Aggregate a session/visit-level DataFrame to one row per user, including all columns.
    Columns are divided into three categories by default:
      1. Numeric (float/int) columns
      2. Datetime columns
      3. Categorical/string columns

    Each category uses a specified "strategy" to aggregate:
      - numeric_strategy  -> {"sum", "mean", "max", "min", "count"} or a custom aggregator
      - date_strategy     -> {"min", "max"} or a custom aggregator
      - categorical_strategy -> {"majority", "unique", "first"} or a custom aggregator

    If you want a **column-specific** aggregator that differs from the default, define it in
    custom_strategies={ col_name: callable }, which overrides the default for that column.

    Optionally, handle a `group_col` that indicates user group assignment:
      - If a user appears in multiple groups, handle_multi_group = {"exclude", "first", "all"}.

    Parameters
    ----------
    df : pd.DataFrame
        Session-level data containing user_id_col and various columns (numeric, date, categorical).
    user_id_col : str
        Column name that identifies unique users (e.g., "fullVisitorId").
    group_col : str, optional
        If provided, column specifying user group assignment (e.g., "test"/"control").
    handle_multi_group : {"exclude", "first", "all"}, default "exclude"
        How to handle users that appear in multiple distinct groups:
          - "exclude": drop such users entirely.
          - "first": keep the earliest group encountered (based on df order).
          - "all": keep multiple rows if the user truly belongs to multiple groups in the data.
            (Often not recommended for pure A/B tests.)
    numeric_strategy : str or callable, default "sum"
        Aggregation for numeric columns if no custom strategy is provided.
        One of {"sum", "mean", "max", "min", "count"} or a custom callable like np.median.
    date_strategy : str or callable, default "min"
        Aggregation for datetime columns if no custom strategy is provided.
        "min" or "max" or a custom callable.
    categorical_strategy : str or callable, default "majority"
        Aggregation for categorical/string columns if no custom strategy is provided.
        - "majority": the most frequent value in that column for that user.
        - "unique": a pipe-joined string of unique values, e.g. "A|B|C".
        - "first": take the first encountered non-null value in df order.
    custom_strategies : dict, optional
        A dictionary {column_name: aggregator} for column-specific logic, e.g.:
          { "deviceCategory": lambda s: s.mode()[0] }
        This overrides the default numeric/date/categorical strategy for that column.
    exclude_columns : list of str, optional
        List of columns to exclude entirely from the aggregation.
    datetime_formats : dict, optional
        For columns that are object/string but actually contain dates, you can specify
        {col_name: format_string} to parse them as datetime before aggregation.
        e.g. { "InvoiceDate": "%Y-%m-%d %H:%M:%S" }

    Returns
    -------
    user_df : pd.DataFrame
        One row per unique user_id_col. All relevant columns are aggregated according to
        their respective strategies.
        If group_col is not None:
          - If handle_multi_group="exclude" or "first", user_df has exactly one group per user.
          - If handle_multi_group="all", some users may appear in multiple rows with different groups.
    """

    df_work = df.copy()

    # 2) Identify columns to exclude
    if exclude_columns:
        df_work.drop(
            columns=[c for c in exclude_columns if c in df_work.columns], inplace=True
        )

    # 0) Parse datetime columns if specified in datetime_formats
    if datetime_formats:
        for col, fmt in datetime_formats.items():
            if col in df_work.columns:
                df_work[col] = pd.to_datetime(df_work[col], format=fmt, errors="coerce")

    # 1) If group_col is provided, handle multi-group users first
    if group_col and handle_multi_group in ("exclude", "first"):
        group_counts = df_work.groupby(user_id_col)[group_col].nunique()
        multi_group_users = group_counts[group_counts > 1].index

        if handle_multi_group == "exclude":
            # remove them from df_work
            df_work = df_work[~df_work[user_id_col].isin(multi_group_users)]

        elif handle_multi_group == "first":
            # We'll keep the earliest group encountered for that user (based on row order).
            # Step A: identify the first group for each user
            # Step B: after we do that, we remove rows that conflict.

            # map from user -> first group
            user_group_map = {}
            for idx, row in df_work.iterrows():
                uid = row[user_id_col]
                if uid not in user_group_map:
                    user_group_map[uid] = row[group_col]

            # filter out any row that doesn't match the assigned group
            def _matches_first_group(row):
                return row[group_col] == user_group_map[row[user_id_col]]

            df_work = df_work[df_work.apply(_matches_first_group, axis=1)]

    # 3) Determine column types for the remaining columns (besides user_id_col, group_col)
    candidate_cols = [c for c in df_work.columns if c not in [user_id_col]]
    if group_col and (group_col in candidate_cols):
        # We handle group_col after numeric/categorical/datetime are aggregated
        candidate_cols.remove(group_col)

    # Separate columns into numeric, datetime, categorical
    numeric_cols = []
    datetime_cols = []
    categorical_cols = []

    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df_work[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df_work[col]):
            datetime_cols.append(col)
        else:
            categorical_cols.append(col)

    # 4) Build an aggregation dictionary
    # If handle_multi_group="all" and group_col is not None, we want to group by [user_id_col, group_col].
    # Otherwise, just group by user_id_col.
    group_keys = [user_id_col]
    if group_col and handle_multi_group == "all":
        group_keys.append(group_col)

    # We'll create an agg_dict per column based on strategies
    # But we rely on a fallback aggregator function that calls the right approach
    all_cols = numeric_cols + datetime_cols + categorical_cols
    agg_functions = {}

    for col in all_cols:
        if custom_strategies and col in custom_strategies:
            agg_functions[col] = custom_strategies[col]
        else:
            if col in numeric_cols:
                agg_functions[col] = _make_numeric_aggregator(numeric_strategy)
            elif col in datetime_cols:
                agg_functions[col] = _make_date_aggregator(date_strategy)
            else:  # categorical
                agg_functions[col] = _make_categorical_aggregator(categorical_strategy)

    # 5) Perform the groupby aggregation
    user_df = df_work.groupby(group_keys, as_index=False).agg(agg_functions)

    # 6) Re-incorporate the group_col if handle_multi_group != "all"
    #    For "exclude" or "first", each user has at most one group, so we can just pick it.
    if group_col and handle_multi_group in ("exclude", "first"):
        # Each user is now unique, so let's get the group from the original df_work
        # after filtering multi-group. We'll just pick the first or unique group
        # for that user
        group_map = df_work.groupby(user_id_col)[group_col].first().to_frame(group_col)
        user_df = user_df.merge(group_map, on=user_id_col, how="left")

    return user_df


# ------------------------------------------------------------------
# Internal helpers to create aggregator functions dynamically
# ------------------------------------------------------------------


def _make_numeric_aggregator(
    strategy: Union[str, Callable]
) -> Callable[[pd.Series], float]:
    """
    Returns a function that aggregates numeric data with the chosen strategy or callable.
    """
    if callable(strategy):
        return strategy

    if strategy == "sum":
        return "sum"
    elif strategy == "mean":
        return "mean"
    elif strategy == "max":
        return "max"
    elif strategy == "min":
        return "min"
    elif strategy == "count":
        # count of non-null entries
        return "count"
    else:
        # fallback, try to interpret the string as a valid aggregator
        return strategy


def _make_date_aggregator(
    strategy: Union[str, Callable]
) -> Callable[[pd.Series], pd.Timestamp]:
    """
    Returns a function/aggregator for datetime columns (min, max, or custom).
    """
    if callable(strategy):
        return strategy

    if strategy == "min":
        return "min"
    elif strategy == "max":
        return "max"
    else:
        # fallback
        return strategy


def _make_categorical_aggregator(
    strategy: Union[str, Callable]
) -> Callable[[pd.Series], str]:
    """
    Returns a function that aggregates categorical/string data:
      - "majority" : most frequent value
      - "unique"   : sorted unique values joined by '|'
      - "first"    : first non-null value
    or a custom callable.
    """
    if callable(strategy):
        return strategy

    def majority_value(s: pd.Series) -> str:
        # mode could have multiple values, pick the first
        mode_vals = s.mode(dropna=True)
        return str(mode_vals.iloc[0]) if not mode_vals.empty else ""

    def unique_values(s: pd.Series) -> str:
        unique_set = sorted(s.dropna().unique())
        return "|".join(str(x) for x in unique_set)

    def first_value(s: pd.Series) -> str:
        for val in s:
            if pd.notnull(val):
                return str(val)
        return ""

    if strategy == "majority":
        return majority_value
    elif strategy == "unique":
        return unique_values
    elif strategy == "first":
        return first_value
    else:
        # fallback
        return strategy
