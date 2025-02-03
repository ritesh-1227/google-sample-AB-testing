import os

# BigQuery configuration
BIGQUERY_PROJECT = os.getenv("BIGQUERY_PROJECT", "googanalyics-staging-project")
# BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "your_dataset_name")

# Other configuration variables (e.g., Streamlit settings)
APP_TITLE = "Google Analytics A/B Testing & Customer Analytics"
