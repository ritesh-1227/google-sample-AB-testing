"""
A comprehensive object-oriented A/B testing module with multiple statistical methods,
transformations, and handling for skewed or zero-inflated data.

Usage Example:
-------------
from ab_testing import ABTest

# Suppose df is your cleaned DataFrame of sessions with columns:
#   trafficSource, totalTransactionRevenue, transactions, timeOnSite, etc.
# Define group logic (for instance, based on trafficSource):
control_filter = {"trafficSource": "google"}
test_filter = {"trafficSource": "Others"}

ab = ABTest(df=df, control_filter=control_filter, test_filter=test_filter)

# Run a parametric t-test on totalTransactionRevenue with a log transform:
result = ab.run_test(
    column="totalTransactionRevenue",
    test_type="t_test",
    transform="log",
    zero_inflation=False
)
print(result)
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class ABTest:
    """
    A robust class for performing A/B tests on potentially skewed or zero-inflated data.
    Provides parametric (t-test), non-parametric (Mann-Whitney), and optional two-part
    zero-inflation handling.

    Attributes
    ----------
    df : pd.DataFrame
        The input DataFrame containing metrics to analyze.
    control_df : pd.DataFrame
        Subset of df corresponding to the control group.
    test_df : pd.DataFrame
        Subset of df corresponding to the test group.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        control_filter: Dict[str, Any],
        test_filter: Dict[str, Any],
        user_id_col: Optional[str] = None,
    ):
        """
        Initialize the ABTest class by splitting the df into control_df and test_df
        based on simple filter logic.

        Parameters
        ----------
        df : pd.DataFrame
            The cleaned data DataFrame (numeric columns should be float for correct handling of NaN).
        control_filter : dict
            A dictionary of {column: value} to filter rows belonging to the control group.
        test_filter : dict
            A dictionary of {column: value} to filter rows belonging to the test group.
        user_id_col : str, optional
            Column name identifying unique users. If provided, you can group or deduplicate
            by user before performing the test. Not used by default.
        """
        self.df = df
        self.control_df = self._filter_df(control_filter)
        self.test_df = self._filter_df(test_filter)
        self.user_id_col = user_id_col

    def _filter_df(self, filter_dict: Dict[str, Any]) -> pd.DataFrame:
        """Return a subset of self.df where each col == val in filter_dict."""
        subset = self.df.copy()
        for col, val in filter_dict.items():
            subset = subset[subset[col] == val]
        return subset

    def run_test(
        self,
        column: str,
        test_type: str = "t_test",
        transform: str = "none",
        zero_inflation: bool = False,
        alpha: float = 0.05,
        add_constant: float = 1.0,
        winsor_percentile: float = 95.0,
        trim_percentile: float = 95.0,
    ) -> Dict[str, Any]:
        """
        Run an A/B test on the specified column using the chosen test (parametric or non-parametric),
        optional transformations, and optional zero-inflation handling.

        Parameters
        ----------
        column : str
            The numeric column on which to perform the A/B test.
        test_type : {"t_test", "mannwhitney", "bayesian_conversions", "bayesian_means"}, default "t_test"
            The type of statistical test to run.
            - "t_test": two-sample t-test (assuming normally distributed data).
            - "mannwhitney": Mann-Whitney U test for median-based comparison.
            - "bayesian_conversions": Beta-Bernoulli approach for 0/1 data (like a conversion rate).
            - "bayesian_means": Simple normal model approach for a continuous metric.
        transform : {"none", "log", "winsor", "trim", "boxcox"}, default "none"
            A transformation to apply to the data prior to testing. Helps mitigate skewness/outliers.
            - "none": no transform
            - "log": apply log(x + add_constant)
            - "winsor": clip extreme values at the given percentile
            - "trim": remove values above the given percentile
            - "boxcox": apply box-cox transform (only for strictly positive data)
        zero_inflation : bool, default False
            If True, we perform a two-part test:
            1) Compare proportion of zeros between control & test (chi-square).
            2) Perform the chosen test_type on non-zero rows only.
        alpha : float, default 0.05
            The significance level for p-value interpretation (mainly for reference).
        add_constant : float, default 1.0
            A small constant added before log transform to avoid log(0).
        winsor_percentile : float, default 95.0
            The percentile at which values are winsorized if transform="winsor".
        trim_percentile : float, default 95.0
            The percentile above which values are removed if transform="trim".

        Returns
        -------
        results : dict
            A dictionary with test details and results (p-value, test statistic, effect sizes).
        """
        results = {
            "test_type": test_type,
            "column": column,
            "transform": transform,
            "zero_inflation": zero_inflation,
            "alpha": alpha,
        }

        # Possibly run a two-part test if zero_inflation is True
        if zero_inflation:
            # 1) Compare proportion of zeros in control vs. test
            zero_test_res = self._compare_zero_proportions(column)
            results["zero_test"] = zero_test_res

            # 2) Filter to non-zero rows for the main test
            control_nonzero = self.control_df[self.control_df[column] > 0]
            test_nonzero = self.test_df[self.test_df[column] > 0]

            # If there's not enough data in non-zero portion, skip main test
            if len(control_nonzero) < 2 or len(test_nonzero) < 2:
                results["main_test"] = {
                    "error": "Not enough non-zero data to run the main test after zero inflation check."
                }
                return results

            # Transform, then run main test
            control_vals = self._apply_transform(
                control_nonzero[column].dropna(),
                transform,
                add_constant,
                winsor_percentile,
                trim_percentile,
            )
            test_vals = self._apply_transform(
                test_nonzero[column].dropna(),
                transform,
                add_constant,
                winsor_percentile,
                trim_percentile,
            )

            main_test_res = self._run_stat_test(control_vals, test_vals, test_type)
            results["main_test"] = main_test_res
            return results

        # If zero_inflation is False, just transform & run the test on the entire data
        control_vals = self._apply_transform(
            self.control_df[column].dropna(),
            transform,
            add_constant,
            winsor_percentile,
            trim_percentile,
        )
        test_vals = self._apply_transform(
            self.test_df[column].dropna(),
            transform,
            add_constant,
            winsor_percentile,
            trim_percentile,
        )

        main_test_res = self._run_stat_test(control_vals, test_vals, test_type)
        results["main_test"] = main_test_res

        return results

    def _run_stat_test(
        self, control_vals: np.ndarray, test_vals: np.ndarray, test_type: str
    ) -> Dict[str, Any]:
        """
        Run the chosen statistical test on the two arrays of data.

        Returns a dictionary with keys: "test_statistic", "p_value", "control_mean",
        "test_mean", etc. (depending on the test).
        """
        if len(control_vals) < 2 or len(test_vals) < 2:
            return {"error": "Insufficient data in control/test for chosen test."}

        # Distinguish test types
        if test_type == "t_test":
            t_stat, p_val = stats.ttest_ind(control_vals, test_vals, nan_policy="omit")
            return {
                "sample_sizes": (len(control_vals), len(test_vals)),
                "test_statistic": t_stat,
                "p_value": p_val,
                "control_mean": float(np.mean(control_vals)),
                "test_mean": float(np.mean(test_vals)),
            }

        elif test_type == "mannwhitney":
            u_stat, p_val = stats.mannwhitneyu(
                control_vals, test_vals, alternative="two-sided"
            )
            return {
                "sample_sizes": (len(control_vals), len(test_vals)),
                "test_statistic": u_stat,
                "p_value": p_val,
                "control_median": float(np.median(control_vals)),
                "test_median": float(np.median(test_vals)),
            }

        elif test_type == "bayesian_conversions":
            # Expect 0/1 data for conversion
            # Beta prior with alpha=1, beta=1 (uniform), combine with data => posterior
            control_sum = float(np.sum(control_vals))
            control_n = float(len(control_vals))
            test_sum = float(np.sum(test_vals))
            test_n = float(len(test_vals))

            # Posterior means
            # alpha_post = alpha_prior + sum_of_ones
            control_alpha = 1.0 + control_sum
            control_beta = 1.0 + (control_n - control_sum)
            test_alpha = 1.0 + test_sum
            test_beta = 1.0 + (test_n - test_sum)

            # Approximate the posterior distributions for the conversion rate
            # Could do Monte Carlo or just return the posterior means
            control_rate = control_alpha / (control_alpha + control_beta)
            test_rate = test_alpha / (test_alpha + test_beta)

            # For an advanced approach, we'd sample from the Beta distributions
            # to estimate P(test > control). Here, let's keep it minimal:
            return {
                "control_conversions": control_sum,
                "control_total": control_n,
                "test_conversions": test_sum,
                "test_total": test_n,
                "control_posterior_mean": control_rate,
                "test_posterior_mean": test_rate,
                "info": "For a complete Bayesian approach, consider posterior sampling.",
            }

        elif test_type == "bayesian_means":
            # Simple normal model approach
            c_mean = float(np.mean(control_vals))
            c_std = float(np.std(control_vals, ddof=1))
            c_n = len(control_vals)

            t_mean = float(np.mean(test_vals))
            t_std = float(np.std(test_vals, ddof=1))
            t_n = len(test_vals)

            # This approach can get quite involved. We'll store basic summary stats,
            # acknowledging that a full Bayesian approach would involve specifying
            # priors, sampling, HPD intervals, etc.
            return {
                "control_mean": c_mean,
                "control_std": c_std,
                "control_n": c_n,
                "test_mean": t_mean,
                "test_std": t_std,
                "test_n": t_n,
                "info": "In-depth Bayesian means testing requires priors & sampling.",
            }

        else:
            return {"error": f"Unsupported test_type: {test_type}"}

    def _apply_transform(
        self,
        values: pd.Series,
        transform: str,
        add_constant: float,
        winsor_p: float,
        trim_p: float,
    ) -> np.ndarray:
        """
        Apply a chosen transformation to a numeric series to mitigate skewness or outliers.
        transform: "none", "log", "winsor", "trim", or "boxcox".
        """
        arr = values.values

        if transform == "none":
            return arr

        elif transform == "log":
            # Ensure no negative or zero
            return np.log(arr + add_constant)

        elif transform == "winsor":
            # Clip outliers above winsor_p
            high = np.percentile(arr, winsor_p)
            low = (
                np.percentile(arr, 100 - winsor_p)
                if winsor_p < 50
                else np.percentile(arr, 0)
            )
            arr = np.clip(arr, low, high)
            return arr

        elif transform == "trim":
            # Remove data above a certain percentile
            cap = np.percentile(arr, trim_p)
            trimmed = arr[arr <= cap]
            return trimmed

        elif transform == "boxcox":
            # Strictly positive data required
            # If any zero or negative, you must offset first
            shift = 0
            if arr.min() <= 0:
                shift = abs(arr.min()) + 1e-9
            from scipy import special

            bc_values, _ = stats.boxcox(arr + shift)
            return bc_values

        return arr

    def _compare_zero_proportions(self, column: str) -> Dict[str, Any]:
        """
        Compare proportion of zeros in control vs. test using a chi-square (2x2) test.

        Returns a dict with 'chi2_stat', 'p_value', 'control_zero_rate', 'test_zero_rate', etc.
        """
        control_zero_count = (self.control_df[column] == 0).sum()
        control_n = len(self.control_df)
        test_zero_count = (self.test_df[column] == 0).sum()
        test_n = len(self.test_df)

        # 2x2 contingency table
        #        Zero   NonZero
        # Ctl    A      B
        # Test   C      D
        A = control_zero_count
        B = control_n - control_zero_count
        C = test_zero_count
        D = test_n - test_zero_count

        chi2, p_val, *_ = stats.chi2_contingency([[A, B], [C, D]])
        return {
            "control_zero_rate": float(A / control_n if control_n > 0 else np.nan),
            "test_zero_rate": float(C / test_n if test_n > 0 else np.nan),
            "chi2_statistic": chi2,
            "p_value": p_val,
            "table": [[int(A), int(B)], [int(C), int(D)]],
        }
