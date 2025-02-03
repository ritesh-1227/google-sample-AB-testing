import os

import pandas as pd

# Adjust working directory so that Python sees 'src' as a package
if not os.path.exists("src"):
    os.chdir("..")

print("Updated working directory:", os.getcwd())

from src.data_cleaning import clean_sessions_data
from src.data_extraction import BigQueryClient
from src.user_aggregation import aggregate_user_data

# 1. Instantiate the BigQuery client
client = BigQueryClient()

# 2. Extract data (without date filtering, limit to e.g. 50,000 rows)
raw_data = client.get_sessions_data(limit=1000000)

# 3. Clean the data
options = {
    "convert_date": False,
    "revenue_adjustment": True,
    "fill_missing_transactions": True,
    "fill_missing_pageviews": True,
    "fill_missing_timeOnSite": True,
    "timeOnSite_zero_floor": True,
}

cleaned_data = clean_sessions_data(raw_data, cleaning_options=options)

user_df = aggregate_user_data(
    df=cleaned_data,
    user_id_col="fullVisitorId",
    group_col=None,  # or "experimentGroup"
    handle_multi_group="first",
    numeric_strategy="mean",  # sum all numeric columns
    date_strategy="min",  # earliest date
    categorical_strategy="majority",  # pick the most frequent category for each user
    custom_strategies={"visitNumber": "max", "transactions": "sum"},
    exclude_columns=["visitId"],  # maybe we don't want visitId in final user-level
    datetime_formats=None,  # if needed, define date formats to parse
)

# 4. Save the cleaned data to disk in the 'data/' folder
os.makedirs("data", exist_ok=True)  # Ensure the folder exists

# Save as CSV
csv_path = os.path.join("data", "user_agg_cleaned.csv")
cleaned_data.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}.")
