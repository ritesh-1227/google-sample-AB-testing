# src/data_extraction.py

import pandas as pd
import pandas_gbq

from src.config import BIGQUERY_PROJECT


class BigQueryClient:
    def __init__(self, project_id: str = BIGQUERY_PROJECT):
        """
        Initialize the BigQuery client using pandas-gbq.
        """
        self.project_id = project_id

    def run_query(self, query: str, dtypes: dict = None) -> pd.DataFrame:
        """
        Execute a SQL query using pandas-gbq and return the result as a Pandas DataFrame.
        """
        try:
            # The dialect is set to 'standard' by default.
            df = pandas_gbq.read_gbq(
                query, project_id=self.project_id, dialect="standard", dtypes=dtypes
            )
            print("Query executed successfully using pandas-gbq.")
            return df
        except Exception as e:
            print("Error running query using pandas-gbq:", e)
            raise

    def get_sessions_data(
        self, start_date: str = None, end_date: str = None, limit: int = 1000000
    ) -> pd.DataFrame:
        """
        Retrieve sessions data from the Google Analytics Sample Store.

        Parameters:
          - start_date: (optional) string in 'YYYYMMDD' format.
          - end_date: (optional) string in 'YYYYMMDD' format.
          - limit: number of records to return (default 1,000,000).

        If no dates are provided, the query will not filter by date.
        """
        base_query = """
            SELECT
              fullVisitorId,
              visitId,
              visitNumber,
              date,
              totals.pageviews AS pageviews,
              totals.timeOnSite AS timeOnSite,
              totals.transactions AS transactions,
              totals.totalTransactionRevenue AS totalTransactionRevenue,
              trafficSource.source AS trafficSource,
              trafficSource.medium AS trafficMedium,
              trafficSource.campaign AS trafficCampaign,
              geoNetwork.country AS country,
              geoNetwork.city AS city
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        """
        conditions = []
        if start_date and end_date:
            conditions.append(f"_TABLE_SUFFIX BETWEEN '{start_date}' AND '{end_date}'")
        if conditions:
            query = base_query + " WHERE " + " AND ".join(conditions)
        else:
            query = base_query

        query += f" LIMIT {limit}"
        print("Final Query:", query)

        dtypes = {
            "fullVisitorId": "string",
            "visitId": "string",
            "visitNumber": "Int64",
            "date": "datetime64[ns]",
            "pageviews": "Int64",
            "timeOnSite": "Int64",
            "transactions": "Int64",
            "totalTransactionRevenue": "Int64",
            "trafficSource": "string",
            "trafficMedium": "string",
            "trafficCampaign": "string",
            "country": "string",
            "city": "string",
        }
        return self.run_query(query, dtypes=dtypes)
