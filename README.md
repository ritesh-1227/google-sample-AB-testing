# Google Analytics A/B Testing & Analytics

A streamlined end-to-end project for data exploration and A/B testing using user-level data (e.g., from Google Analytics).

## Key Features

- **Data Aggregation**: Convert session-level data to user-level for easier analysis.
- **Exploratory Data Analysis**: Histograms, funnel visualization, campaign and country-level insights.
- **Hypothesis Testing**:
  - Product Recommendation: Random control vs. test assignment.
  - Dynamic Pricing: Threshold-based assignment to control/test.
- **Interactive Streamlit App**: Upload or load a local CSV, run EDA, configure A/B tests, and see results.


# Project Structure

```google_analytics_ab_testing/

├── streamlit_app.py          # Main Streamlit UI
├── data/
│   └── user_level.csv       # Example local dataset
├── src/
│   ├── __init__.py          # Enables src as a package
│   ├── ab_testing.py        # Core A/B testing logic
│   ├── ab_test_reporting.py # User-friendly output for test results
│   ├── hypothesis_recommendation.py
│   ├── hypothesis_pricing.py
│   └── data_aggregation.py  # Example script for user-level aggregation
└── environment.yml          # Conda environment (optional)

```

## Installation & Setup

1. **Clone** or **download** this repository.
2. *(Optional)* Create a Python environment (e.g., via `conda env create -f environment.yml`).
3. **Activate** your environment:

   ```bash
   conda activate google_ab_testing

## Usage

Launch the Streamlit app from the project root:

  ```bash
  streamlit run app/streamlit_app.py
