import pandas as pd

from user_aggregation import aggregate_user_data

# Suppose we have session-level data with columns:
#  fullVisitorId (user), visitId, date, transactions, totalTransactionRevenue,
#  deviceCategory, country, city, and so on.

df_sessions = pd.read_csv("data/cleaned_sessions.csv")

# We want a user-level table with sum of numeric columns, earliest date, and most frequent category
user_df = aggregate_user_data(
    df=df_sessions,
    user_id_col="fullVisitorId",
    group_col=None,  # or "experimentGroup"
    handle_multi_group="first",
    numeric_strategy="median",  # sum all numeric columns
    date_strategy="min",  # earliest date
    categorical_strategy="majority",  # pick the most frequent category for each user
    custom_strategies={
        "city": "unique"  # for the "city" column, store all unique visited cities
    },
    exclude_columns=["visitId"],  # maybe we don't want visitId in final user-level
    datetime_formats=None,  # if needed, define date formats to parse
)

print(user_df.head(5))

user_df.to_csv("data/user_agg_data.csv")
