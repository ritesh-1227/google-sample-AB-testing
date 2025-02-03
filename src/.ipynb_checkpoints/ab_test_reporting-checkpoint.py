# src/ab_test_reporting.py

from typing import Dict


def interpret_ab_results(results: Dict) -> str:
    """
    Provide a user-friendly explanation for the A/B test results returned by ABTest.
    This includes transformations, zero-inflation details, test type, p-values, etc.
    """
    column = results.get("column", "unknown_metric")
    test_type = results.get("test_type", "unknown_test")
    transform = results.get("transform", "none")
    zero_inflation = results.get("zero_inflation", False)
    alpha = results.get("alpha", 0.05)

    text = [
        f"A/B Test on '{column}' using '{test_type}' test.",
        f"Applied transform='{transform}', zero_inflation={zero_inflation}. (alpha={alpha})",
    ]

    # If zero_inflation was used, look for 'zero_test'
    if zero_inflation and "zero_test" in results:
        zt = results["zero_test"]
        text.append(
            f"Zero-proportion test: control_zero_rate={zt.get('control_zero_rate', float('nan')):.2%}, "
            f"test_zero_rate={zt.get('test_zero_rate', float('nan')):.2%}, p-value={zt.get('p_value', float('nan')):.3f}"
        )

    main_test = results.get("main_test", {})
    if "error" in main_test:
        text.append(f"Main Test Error: {main_test['error']}")
        return "\n".join(text)

    # Distinguish test types
    if test_type == "t_test":
        p_val = main_test.get("p_value", None)
        ctrl_mean = main_test.get("control_mean", float("nan"))
        tst_mean = main_test.get("test_mean", float("nan"))
        text.append(
            f"Control Mean={ctrl_mean:.2f}, Test Mean={tst_mean:.2f}, p-value={p_val:.4f}"
        )
        if p_val is not None and p_val < alpha:
            text.append("Statistically significant difference.")
        else:
            text.append("No statistically significant difference detected.")

    elif test_type == "mannwhitney":
        p_val = main_test.get("p_value", None)
        ctrl_med = main_test.get("control_median", float("nan"))
        tst_med = main_test.get("test_median", float("nan"))
        text.append(
            f"Control Median={ctrl_med:.2f}, Test Median={tst_med:.2f}, p-value={p_val:.4f}"
        )
        if p_val is not None and p_val < alpha:
            text.append("Statistically significant difference.")
        else:
            text.append("No statistically significant difference detected.")

    elif test_type == "bayesian_conversions":
        crtl_pm = main_test.get("control_posterior_mean", float("nan"))
        test_pm = main_test.get("test_posterior_mean", float("nan"))
        text.append(
            f"Bayesian Beta-Bernoulli: control_mean={crtl_pm:.3f}, test_mean={test_pm:.3f}"
        )
        text.append("For deeper inference, consider posterior sampling (not shown).")

    elif test_type == "bayesian_means":
        cmean = main_test.get("control_mean", float("nan"))
        tmean = main_test.get("test_mean", float("nan"))
        text.append(
            f"Bayesian normal approach (basic): control_mean={cmean:.2f}, test_mean={tmean:.2f}"
        )
        text.append(main_test.get("info", "No additional info."))

    else:
        text.append(f"Unsupported or unknown test type: {test_type}")

    return "\n".join(text)
