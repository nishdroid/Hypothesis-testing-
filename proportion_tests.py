import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .statistical_utils import StatisticalUtils
from .visualization_utils import VisualizationUtils

class ProportionTests:
    def __init__(self):
        self.stat_utils = StatisticalUtils()
        self.viz_utils = VisualizationUtils()
    
    def render(self, confidence_level, bootstrap_iterations):
        st.header("📊 Proportion Tests")
        st.markdown("Comprehensive testing for proportions, rates, and categorical data analysis.")
        
        # Test selection
        test_type = st.selectbox(
            "Select Proportion Test:",
            [
                "One-sample proportion test",
                "Two-sample proportion test",
                "Chi-square goodness of fit",
                "Chi-square test of independence",
                "Fisher's exact test",
                "McNemar's test",
                "Cochran's Q test",
                "Bootstrap proportion tests"
            ]
        )
        
        if test_type == "One-sample proportion test":
            self._one_sample_proportion(confidence_level, bootstrap_iterations)
        elif test_type == "Two-sample proportion test":
            self._two_sample_proportion(confidence_level, bootstrap_iterations)
        elif test_type == "Chi-square goodness of fit":
            self._chi_square_goodness_of_fit(confidence_level)
        elif test_type == "Chi-square test of independence":
            self._chi_square_independence(confidence_level)
        elif test_type == "Fisher's exact test":
            self._fishers_exact_test(confidence_level)
        elif test_type == "McNemar's test":
            self._mcnemars_test(confidence_level)
        elif test_type == "Cochran's Q test":
            self._cochrans_q_test(confidence_level)
        elif test_type == "Bootstrap proportion tests":
            self._bootstrap_proportion_tests(confidence_level, bootstrap_iterations)
    
    def _one_sample_proportion(self, confidence_level, bootstrap_iterations):
        """One-sample proportion test (binomial test)"""
        st.subheader("One-Sample Proportion Test")
        st.markdown("Tests whether an observed proportion differs significantly from an expected value.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Input Data")
            
            input_method = st.radio(
                "Input method:",
                ["Count and Total", "Raw Binary Data", "Proportion and Sample Size"]
            )
            
            if input_method == "Count and Total":
                successes = st.number_input("Number of successes:", min_value=0, value=15)
                n_trials = st.number_input("Total trials:", min_value=1, value=50)
                observed_prop = successes / n_trials if n_trials > 0 else 0
            elif input_method == "Raw Binary Data":
                binary_input = st.text_area(
                    "Enter binary data (1s and 0s):",
                    placeholder="1,0,1,1,0,1,0,0,1,1",
                    height=100
                )
                if binary_input:
                    try:
                        binary_data = [int(x.strip()) for x in binary_input.split(',')]
                        successes = sum(binary_data)
                        n_trials = len(binary_data)
                        observed_prop = successes / n_trials
                    except:
                        st.error("Invalid binary data format")
                        return
                else:
                    successes = n_trials = observed_prop = 0
            else:  # Proportion and Sample Size
                observed_prop = st.number_input("Observed proportion:", min_value=0.0, max_value=1.0, value=0.3)
                n_trials = st.number_input("Sample size:", min_value=1, value=50)
                successes = int(observed_prop * n_trials)
            
            expected_prop = st.number_input("Expected proportion (H₀):", min_value=0.0, max_value=1.0, value=0.5)
            
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"],
                key="one_prop_alt"
            )
        
        with col1:
            if n_trials == 0:
                st.warning("Please provide valid data")
                return
            
            # Display current data
            st.subheader("Data Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Successes", successes)
                st.metric("Sample Size", n_trials)
            
            with summary_col2:
                st.metric("Observed Proportion", f"{observed_prop:.4f}")
                st.metric("Expected Proportion", f"{expected_prop:.4f}")
            
            with summary_col3:
                failures = n_trials - successes
                st.metric("Failures", failures)
                st.metric("Difference", f"{observed_prop - expected_prop:.4f}")
            
            # Check assumptions
            st.subheader("Assumption Checking")
            
            # Sample size requirements
            expected_successes = n_trials * expected_prop
            expected_failures = n_trials * (1 - expected_prop)
            
            assumption_col1, assumption_col2 = st.columns(2)
            
            with assumption_col1:
                st.write("**Normal Approximation:**")
                if expected_successes >= 5 and expected_failures >= 5:
                    st.success("✅ np ≥ 5 and n(1-p) ≥ 5")
                    use_normal = True
                else:
                    st.warning("❌ Small sample - use exact binomial test")
                    use_normal = False
                
                st.write(f"Expected successes: {expected_successes:.1f}")
                st.write(f"Expected failures: {expected_failures:.1f}")
            
            with assumption_col2:
                st.write("**Independence:**")
                st.info("Assume observations are independent")
                if n_trials > 100:
                    st.write("✅ Large sample supports independence assumption")
                else:
                    st.write("⚠️ Ensure sampling method supports independence")
            
            # Perform tests
            st.subheader("Test Results")
            
            # Exact binomial test
            if alternative == "two-sided":
                p_value_exact = 2 * min(
                    stats.binom.cdf(successes, n_trials, expected_prop),
                    1 - stats.binom.cdf(successes - 1, n_trials, expected_prop)
                )
            elif alternative == "greater":
                p_value_exact = 1 - stats.binom.cdf(successes - 1, n_trials, expected_prop)
            else:  # less
                p_value_exact = stats.binom.cdf(successes, n_trials, expected_prop)
            
            # Normal approximation (z-test)
            if use_normal:
                se = np.sqrt(expected_prop * (1 - expected_prop) / n_trials)
                z_stat = (observed_prop - expected_prop) / se
                
                if alternative == "two-sided":
                    p_value_normal = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                elif alternative == "greater":
                    p_value_normal = 1 - stats.norm.cdf(z_stat)
                else:  # less
                    p_value_normal = stats.norm.cdf(z_stat)
            
            # Display results
            test_col1, test_col2 = st.columns(2)
            
            with test_col1:
                st.write("**Exact Binomial Test:**")
                st.metric("p-value (exact)", f"{p_value_exact:.6f}")
                
                alpha = 1 - confidence_level
                if p_value_exact < alpha:
                    st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                else:
                    st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
            
            with test_col2:
                if use_normal:
                    st.write("**Normal Approximation (z-test):**")
                    st.metric("z-statistic", f"{z_stat:.4f}")
                    st.metric("p-value (normal)", f"{p_value_normal:.6f}")
                    
                    if p_value_normal < alpha:
                        st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                    else:
                        st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                else:
                    st.info("Normal approximation not appropriate for small samples")
            
            # Confidence intervals
            st.subheader("Confidence Intervals")
            
            # Exact confidence interval (Clopper-Pearson)
            if successes == 0:
                ci_lower_exact = 0
            else:
                ci_lower_exact = stats.beta.ppf((1-confidence_level)/2, successes, n_trials - successes + 1)
            
            if successes == n_trials:
                ci_upper_exact = 1
            else:
                ci_upper_exact = stats.beta.ppf(1-(1-confidence_level)/2, successes + 1, n_trials - successes)
            
            # Wilson confidence interval
            z_critical = stats.norm.ppf(1 - (1-confidence_level)/2)
            wilson_center = (observed_prop + z_critical**2 / (2 * n_trials)) / (1 + z_critical**2 / n_trials)
            wilson_margin = z_critical * np.sqrt(observed_prop * (1 - observed_prop) / n_trials + z_critical**2 / (4 * n_trials**2)) / (1 + z_critical**2 / n_trials)
            ci_lower_wilson = wilson_center - wilson_margin
            ci_upper_wilson = wilson_center + wilson_margin
            
            ci_col1, ci_col2 = st.columns(2)
            
            with ci_col1:
                st.write("**Exact CI (Clopper-Pearson):**")
                st.write(f"[{ci_lower_exact:.4f}, {ci_upper_exact:.4f}]")
                
                if expected_prop < ci_lower_exact or expected_prop > ci_upper_exact:
                    st.write("✅ Expected proportion outside CI")
                else:
                    st.write("❌ Expected proportion within CI")
            
            with ci_col2:
                st.write("**Wilson Confidence Interval:**")
                st.write(f"[{ci_lower_wilson:.4f}, {ci_upper_wilson:.4f}]")
                
                if expected_prop < ci_lower_wilson or expected_prop > ci_upper_wilson:
                    st.write("✅ Expected proportion outside CI")
                else:
                    st.write("❌ Expected proportion within CI")
            
            # Effect size
            st.subheader("Effect Size")
            
            # Cohen's h for proportions
            cohens_h = 2 * (np.arcsin(np.sqrt(observed_prop)) - np.arcsin(np.sqrt(expected_prop)))
            
            effect_col1, effect_col2 = st.columns(2)
            
            with effect_col1:
                st.metric("Cohen's h", f"{cohens_h:.4f}")
                
                if abs(cohens_h) < 0.2:
                    effect_magnitude = "small"
                elif abs(cohens_h) < 0.5:
                    effect_magnitude = "medium"
                else:
                    effect_magnitude = "large"
                
                st.write(f"**Effect size:** {effect_magnitude}")
            
            with effect_col2:
                # Odds ratio
                if observed_prop > 0 and observed_prop < 1 and expected_prop > 0 and expected_prop < 1:
                    observed_odds = observed_prop / (1 - observed_prop)
                    expected_odds = expected_prop / (1 - expected_prop)
                    odds_ratio = observed_odds / expected_odds
                    st.metric("Odds Ratio", f"{odds_ratio:.4f}")
                else:
                    st.info("Cannot calculate odds ratio (proportion = 0 or 1)")
            
            # Visualization
            fig = self.viz_utils.create_proportion_test_plot(
                successes, n_trials, expected_prop, observed_prop, confidence_level
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _two_sample_proportion(self, confidence_level, bootstrap_iterations):
        """Two-sample proportion test"""
        st.subheader("Two-Sample Proportion Test")
        st.markdown("Compare proportions between two independent groups.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Input Data")
            
            input_method = st.radio(
                "Input method:",
                ["Count Data", "Summary Statistics"]
            )
            
            if input_method == "Count Data":
                st.write("**Group 1:**")
                x1 = st.number_input("Successes in Group 1:", min_value=0, value=20)
                n1 = st.number_input("Total in Group 1:", min_value=1, value=100)
                
                st.write("**Group 2:**")
                x2 = st.number_input("Successes in Group 2:", min_value=0, value=15)
                n2 = st.number_input("Total in Group 2:", min_value=1, value=80)
                
                p1 = x1 / n1 if n1 > 0 else 0
                p2 = x2 / n2 if n2 > 0 else 0
                
            else:  # Summary Statistics
                st.write("**Group 1:**")
                p1 = st.number_input("Proportion in Group 1:", min_value=0.0, max_value=1.0, value=0.2)
                n1 = st.number_input("Sample size Group 1:", min_value=1, value=100)
                x1 = int(p1 * n1)
                
                st.write("**Group 2:**")
                p2 = st.number_input("Proportion in Group 2:", min_value=0.0, max_value=1.0, value=0.15)
                n2 = st.number_input("Sample size Group 2:", min_value=1, value=80)
                x2 = int(p2 * n2)
            
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"],
                key="two_prop_alt"
            )
            
            pooled_variance = st.checkbox("Use pooled variance", value=True)
        
        with col1:
            # Data summary
            st.subheader("Data Summary")
            
            summary_data = pd.DataFrame({
                'Group': ['Group 1', 'Group 2', 'Total'],
                'Successes': [x1, x2, x1 + x2],
                'Total': [n1, n2, n1 + n2],
                'Proportion': [f"{p1:.4f}", f"{p2:.4f}", f"{(x1+x2)/(n1+n2):.4f}"],
                'Failures': [n1-x1, n2-x2, (n1+n2)-(x1+x2)]
            })
            
            st.dataframe(summary_data, hide_index=True)
            
            # Check assumptions
            st.subheader("Assumption Checking")
            
            # Sample size requirements
            pooled_p = (x1 + x2) / (n1 + n2)
            expected_cells = [n1 * pooled_p, n1 * (1 - pooled_p), 
                            n2 * pooled_p, n2 * (1 - pooled_p)]
            
            assumption_col1, assumption_col2 = st.columns(2)
            
            with assumption_col1:
                st.write("**Sample Size Requirements:**")
                if all(cell >= 5 for cell in expected_cells):
                    st.success("✅ All expected counts ≥ 5")
                    assumptions_met = True
                else:
                    st.warning("❌ Some expected counts < 5")
                    st.write("Consider Fisher's exact test")
                    assumptions_met = False
                
                st.write(f"Expected counts: {[f'{x:.1f}' for x in expected_cells]}")
            
            with assumption_col2:
                st.write("**Independence:**")
                st.info("Assume independent samples and observations")
                if n1 + n2 > 100:
                    st.write("✅ Large combined sample")
                else:
                    st.write("⚠️ Ensure adequate sample sizes")
            
            # Perform test
            st.subheader("Test Results")
            
            # Two-proportion z-test
            if pooled_variance:
                # Pooled standard error
                se_pooled = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
                z_stat = (p1 - p2) / se_pooled
            else:
                # Unpooled standard error
                se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
                z_stat = (p1 - p2) / se_unpooled
            
            # P-value
            if alternative == "two-sided":
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            elif alternative == "greater":
                p_value = 1 - stats.norm.cdf(z_stat)
            else:  # less
                p_value = stats.norm.cdf(z_stat)
            
            # Chi-square test (equivalent for two-sided)
            contingency_table = np.array([[x1, n1-x1], [x2, n2-x2]])
            chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency_table, correction=False)
            
            # Display results
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("z-statistic", f"{z_stat:.4f}")
                st.metric("p-value (z-test)", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("χ² statistic", f"{chi2_stat:.4f}")
                st.metric("p-value (χ²)", f"{chi2_p:.6f}")
            
            with results_col3:
                st.metric("Difference (p₁ - p₂)", f"{p1 - p2:.4f}")
                alpha = 1 - confidence_level
                if p_value < alpha:
                    st.success("✅ Reject H₀")
                else:
                    st.info("❌ Fail to reject H₀")
            
            # Confidence intervals
            st.subheader("Confidence Intervals")
            
            # CI for difference in proportions
            z_critical = stats.norm.ppf(1 - (1-confidence_level)/2)
            
            if pooled_variance:
                # For hypothesis testing (pooled)
                margin_pooled = z_critical * se_pooled
                ci_lower_pooled = (p1 - p2) - margin_pooled
                ci_upper_pooled = (p1 - p2) + margin_pooled
            
            # For estimation (unpooled) - more appropriate for CI
            se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            margin_unpooled = z_critical * se_unpooled
            ci_lower_unpooled = (p1 - p2) - margin_unpooled
            ci_upper_unpooled = (p1 - p2) + margin_unpooled
            
            ci_col1, ci_col2 = st.columns(2)
            
            with ci_col1:
                st.write("**CI for Difference (Unpooled):**")
                st.write(f"[{ci_lower_unpooled:.4f}, {ci_upper_unpooled:.4f}]")
                
                if 0 < ci_lower_unpooled or 0 > ci_upper_unpooled:
                    st.write("✅ Zero not in confidence interval")
                else:
                    st.write("❌ Zero within confidence interval")
            
            with ci_col2:
                # Individual proportion CIs
                st.write("**Individual Proportion CIs:**")
                
                # Wilson CIs for individual proportions
                wilson1_center = (p1 + z_critical**2 / (2 * n1)) / (1 + z_critical**2 / n1)
                wilson1_margin = z_critical * np.sqrt(p1 * (1 - p1) / n1 + z_critical**2 / (4 * n1**2)) / (1 + z_critical**2 / n1)
                
                wilson2_center = (p2 + z_critical**2 / (2 * n2)) / (1 + z_critical**2 / n2)
                wilson2_margin = z_critical * np.sqrt(p2 * (1 - p2) / n2 + z_critical**2 / (4 * n2**2)) / (1 + z_critical**2 / n2)
                
                st.write(f"Group 1: [{wilson1_center - wilson1_margin:.4f}, {wilson1_center + wilson1_margin:.4f}]")
                st.write(f"Group 2: [{wilson2_center - wilson2_margin:.4f}, {wilson2_center + wilson2_margin:.4f}]")
            
            # Effect sizes
            st.subheader("Effect Sizes")
            
            effect_col1, effect_col2, effect_col3 = st.columns(3)
            
            with effect_col1:
                # Cohen's h
                cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
                st.metric("Cohen's h", f"{cohens_h:.4f}")
                
                if abs(cohens_h) < 0.2:
                    h_magnitude = "small"
                elif abs(cohens_h) < 0.5:
                    h_magnitude = "medium"
                else:
                    h_magnitude = "large"
                st.write(f"Magnitude: {h_magnitude}")
            
            with effect_col2:
                # Relative risk
                if p2 > 0:
                    relative_risk = p1 / p2
                    st.metric("Relative Risk", f"{relative_risk:.4f}")
                    
                    if relative_risk > 1:
                        st.write("Group 1 has higher risk")
                    elif relative_risk < 1:
                        st.write("Group 2 has higher risk")
                    else:
                        st.write("Equal risk")
                else:
                    st.info("Cannot calculate RR (p₂ = 0)")
            
            with effect_col3:
                # Odds ratio
                if p1 > 0 and p1 < 1 and p2 > 0 and p2 < 1:
                    odds1 = p1 / (1 - p1)
                    odds2 = p2 / (1 - p2)
                    odds_ratio = odds1 / odds2
                    st.metric("Odds Ratio", f"{odds_ratio:.4f}")
                    
                    # Log odds ratio CI
                    log_or = np.log(odds_ratio)
                    se_log_or = np.sqrt(1/x1 + 1/(n1-x1) + 1/x2 + 1/(n2-x2))
                    log_ci_lower = log_or - z_critical * se_log_or
                    log_ci_upper = log_or + z_critical * se_log_or
                    
                    st.write(f"95% CI: [{np.exp(log_ci_lower):.4f}, {np.exp(log_ci_upper):.4f}]")
                else:
                    st.info("Cannot calculate OR")
            
            # Visualization
            fig = self.viz_utils.create_two_proportion_plot(
                x1, n1, x2, n2, p1, p2
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _chi_square_goodness_of_fit(self, confidence_level):
        """Chi-square goodness of fit test"""
        st.subheader("Chi-Square Goodness of Fit Test")
        st.markdown("Tests whether observed frequencies match expected distribution.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Input Data")
            
            input_method = st.radio(
                "Input method:",
                ["Manual Entry", "Upload Data"]
            )
            
            if input_method == "Manual Entry":
                n_categories = st.number_input("Number of categories:", min_value=2, max_value=20, value=4)
                
                st.write("**Observed Frequencies:**")
                observed = []
                for i in range(n_categories):
                    freq = st.number_input(f"Category {i+1}:", min_value=0, value=25, key=f"obs_{i}")
                    observed.append(freq)
                
                st.write("**Expected Frequencies:**")
                expected_method = st.radio(
                    "Expected frequencies:",
                    ["Equal", "Proportional", "Manual"]
                )
                
                if expected_method == "Equal":
                    total_obs = sum(observed)
                    expected = [total_obs / n_categories] * n_categories
                elif expected_method == "Proportional":
                    st.write("Enter expected proportions (must sum to 1):")
                    proportions = []
                    for i in range(n_categories):
                        prop = st.number_input(f"Proportion {i+1}:", min_value=0.0, max_value=1.0, 
                                             value=1/n_categories, key=f"prop_{i}")
                        proportions.append(prop)
                    
                    if abs(sum(proportions) - 1.0) > 0.001:
                        st.warning("Proportions should sum to 1.0")
                    
                    total_obs = sum(observed)
                    expected = [prop * total_obs for prop in proportions]
                else:  # Manual
                    expected = []
                    for i in range(n_categories):
                        exp_freq = st.number_input(f"Expected {i+1}:", min_value=0.01, 
                                                 value=25.0, key=f"exp_{i}")
                        expected.append(exp_freq)
                
                categories = [f"Category {i+1}" for i in range(n_categories)]
            
            else:  # Upload Data
                st.info("Upload CSV with 'category' and 'frequency' columns")
                # For now, use manual entry as primary method
                st.warning("Manual entry required for this implementation")
                return
        
        with col1:
            if len(observed) == 0 or len(expected) == 0:
                st.warning("Please enter observed and expected frequencies")
                return
            
            observed = np.array(observed)
            expected = np.array(expected)
            
            # Data summary
            st.subheader("Data Summary")
            
            summary_df = pd.DataFrame({
                'Category': categories,
                'Observed': observed,
                'Expected': expected,
                'Difference': observed - expected,
                'Contribution': (observed - expected)**2 / expected
            })
            
            st.dataframe(summary_df, hide_index=True)
            
            # Check assumptions
            st.subheader("Assumption Checking")
            
            assumption_col1, assumption_col2 = st.columns(2)
            
            with assumption_col1:
                st.write("**Expected Frequency Rule:**")
                min_expected = np.min(expected)
                cells_less_than_5 = np.sum(expected < 5)
                
                if min_expected >= 5:
                    st.success("✅ All expected frequencies ≥ 5")
                elif min_expected >= 1 and cells_less_than_5 <= 0.2 * len(expected):
                    st.warning("⚠️ Some cells < 5, but < 20% of cells")
                else:
                    st.error("❌ Too many cells with expected frequency < 5")
                
                st.write(f"Minimum expected frequency: {min_expected:.2f}")
                st.write(f"Cells with expected < 5: {cells_less_than_5}")
            
            with assumption_col2:
                st.write("**Independence:**")
                st.info("Assume observations are independent")
                st.write("**Sample Size:**")
                total_n = np.sum(observed)
                st.write(f"Total observations: {total_n}")
                if total_n >= 30:
                    st.success("✅ Adequate sample size")
                else:
                    st.warning("⚠️ Small sample size")
            
            # Perform test
            st.subheader("Test Results")
            
            # Chi-square test
            chi2_stat = np.sum((observed - expected)**2 / expected)
            df = len(observed) - 1
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
            
            # Cramer's V (effect size)
            cramers_v = np.sqrt(chi2_stat / np.sum(observed))
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("χ² statistic", f"{chi2_stat:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Degrees of freedom", df)
                st.metric("Cramer's V", f"{cramers_v:.4f}")
            
            with results_col3:
                alpha = 1 - confidence_level
                critical_value = stats.chi2.ppf(confidence_level, df)
                st.metric("Critical value", f"{critical_value:.4f}")
                
                if p_value < alpha:
                    st.success("✅ Reject H₀")
                    st.write("Observed ≠ Expected")
                else:
                    st.info("❌ Fail to reject H₀")
                    st.write("Observed = Expected")
            
            # Effect size interpretation
            st.subheader("Effect Size Interpretation")
            if cramers_v < 0.1:
                effect_magnitude = "negligible"
            elif cramers_v < 0.3:
                effect_magnitude = "small"
            elif cramers_v < 0.5:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            
            st.write(f"**Cramer's V magnitude:** {effect_magnitude}")
            
            # Post-hoc analysis
            if p_value < alpha:
                st.subheader("Post-hoc Analysis")
                st.write("**Standardized Residuals:**")
                
                std_residuals = (observed - expected) / np.sqrt(expected)
                
                residual_df = pd.DataFrame({
                    'Category': categories,
                    'Standardized Residual': std_residuals,
                    'Interpretation': [
                        "Higher than expected" if r > 2 else
                        "Lower than expected" if r < -2 else
                        "As expected" for r in std_residuals
                    ]
                })
                
                st.dataframe(residual_df, hide_index=True)
                
                st.write("**Guidelines:** |residual| > 2 suggests significant deviation")
            
            # Visualization
            fig = self.viz_utils.create_goodness_of_fit_plot(
                categories, observed, expected
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _chi_square_independence(self, confidence_level):
        """Chi-square test of independence"""
        st.subheader("Chi-Square Test of Independence")
        st.markdown("Tests whether two categorical variables are independent.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Input Contingency Table")
            
            # Table dimensions
            n_rows = st.number_input("Number of rows:", min_value=2, max_value=10, value=2)
            n_cols = st.number_input("Number of columns:", min_value=2, max_value=10, value=2)
            
            # Variable names
            row_var = st.text_input("Row variable name:", value="Variable A")
            col_var = st.text_input("Column variable name:", value="Variable B")
            
            st.write("**Enter contingency table values:**")
            
            # Create input matrix
            contingency_data = []
            for i in range(n_rows):
                row_data = []
                st.write(f"Row {i+1}:")
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    with cols[j]:
                        value = st.number_input(f"({i+1},{j+1})", min_value=0, 
                                              value=25, key=f"cell_{i}_{j}")
                        row_data.append(value)
                contingency_data.append(row_data)
            
            contingency_table = np.array(contingency_data)
        
        with col1:
            if contingency_table.size == 0:
                st.warning("Please enter contingency table data")
                return
            
            # Display contingency table
            st.subheader("Contingency Table")
            
            # Create display dataframe with margins
            display_table = pd.DataFrame(contingency_table)
            display_table.columns = [f"Col {j+1}" for j in range(n_cols)]
            display_table.index = [f"Row {i+1}" for i in range(n_rows)]
            
            # Add marginal totals
            display_table['Row Total'] = display_table.sum(axis=1)
            col_totals = display_table.sum(axis=0)
            display_table.loc['Col Total'] = col_totals
            
            st.dataframe(display_table)
            
            # Check assumptions
            st.subheader("Assumption Checking")
            
            # Expected frequencies
            row_totals = contingency_table.sum(axis=1)
            col_totals = contingency_table.sum(axis=0)
            grand_total = contingency_table.sum()
            
            expected_freq = np.outer(row_totals, col_totals) / grand_total
            
            assumption_col1, assumption_col2 = st.columns(2)
            
            with assumption_col1:
                st.write("**Expected Frequency Rule:**")
                min_expected = np.min(expected_freq)
                cells_less_than_5 = np.sum(expected_freq < 5)
                total_cells = expected_freq.size
                
                if min_expected >= 5:
                    st.success("✅ All expected frequencies ≥ 5")
                elif min_expected >= 1 and cells_less_than_5 <= 0.2 * total_cells:
                    st.warning("⚠️ Some cells < 5, but < 20% of cells")
                else:
                    st.error("❌ Too many cells with expected frequency < 5")
                    st.write("Consider Fisher's exact test or combine categories")
                
                st.write(f"Minimum expected: {min_expected:.2f}")
                st.write(f"Cells < 5: {cells_less_than_5}/{total_cells}")
            
            with assumption_col2:
                st.write("**Independence:**")
                st.info("Assume independent observations")
                st.write("**Sample Size:**")
                st.write(f"Total observations: {grand_total}")
                if grand_total >= 50:
                    st.success("✅ Adequate sample size")
                else:
                    st.warning("⚠️ Small sample size")
            
            # Expected frequencies table
            st.subheader("Expected Frequencies")
            expected_df = pd.DataFrame(expected_freq)
            expected_df.columns = [f"Col {j+1}" for j in range(n_cols)]
            expected_df.index = [f"Row {i+1}" for i in range(n_rows)]
            st.dataframe(expected_df.round(2))
            
            # Perform test
            st.subheader("Test Results")
            
            # Chi-square test
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Effect sizes
            cramers_v = np.sqrt(chi2_stat / (grand_total * (min(n_rows, n_cols) - 1)))
            phi_coefficient = np.sqrt(chi2_stat / grand_total) if n_rows == 2 and n_cols == 2 else None
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("χ² statistic", f"{chi2_stat:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Degrees of freedom", dof)
                st.metric("Cramer's V", f"{cramers_v:.4f}")
            
            with results_col3:
                alpha = 1 - confidence_level
                critical_value = stats.chi2.ppf(confidence_level, dof)
                st.metric("Critical value", f"{critical_value:.4f}")
                
                if p_value < alpha:
                    st.success("✅ Reject H₀")
                    st.write("Variables are dependent")
                else:
                    st.info("❌ Fail to reject H₀")
                    st.write("Variables are independent")
            
            # Additional effect sizes for 2x2 tables
            if n_rows == 2 and n_cols == 2:
                st.subheader("2×2 Table Analysis")
                
                effect_col1, effect_col2 = st.columns(2)
                
                with effect_col1:
                    st.metric("Phi coefficient", f"{phi_coefficient:.4f}")
                    
                    # Odds ratio
                    a, b, c, d = contingency_table[0,0], contingency_table[0,1], contingency_table[1,0], contingency_table[1,1]
                    if b > 0 and c > 0:
                        odds_ratio = (a * d) / (b * c)
                        st.metric("Odds Ratio", f"{odds_ratio:.4f}")
                        
                        # OR confidence interval
                        log_or = np.log(odds_ratio)
                        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
                        z_critical = stats.norm.ppf(1 - (1-confidence_level)/2)
                        ci_lower = np.exp(log_or - z_critical * se_log_or)
                        ci_upper = np.exp(log_or + z_critical * se_log_or)
                        st.write(f"OR 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                    else:
                        st.info("Cannot calculate OR (zero cell)")
                
                with effect_col2:
                    # Risk measures
                    risk1 = a / (a + b)
                    risk2 = c / (c + d)
                    
                    if risk2 > 0:
                        relative_risk = risk1 / risk2
                        st.metric("Relative Risk", f"{relative_risk:.4f}")
                    
                    risk_diff = risk1 - risk2
                    st.metric("Risk Difference", f"{risk_diff:.4f}")
            
            # Effect size interpretation
            st.subheader("Effect Size Interpretation")
            if cramers_v < 0.1:
                effect_magnitude = "negligible"
            elif cramers_v < 0.3:
                effect_magnitude = "small"
            elif cramers_v < 0.5:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            
            st.write(f"**Cramer's V magnitude:** {effect_magnitude}")
            
            # Standardized residuals
            if p_value < alpha:
                st.subheader("Standardized Residuals")
                std_residuals = (contingency_table - expected) / np.sqrt(expected)
                
                residuals_df = pd.DataFrame(std_residuals)
                residuals_df.columns = [f"Col {j+1}" for j in range(n_cols)]
                residuals_df.index = [f"Row {i+1}" for i in range(n_rows)]
                
                st.dataframe(residuals_df.round(3))
                st.write("**Guidelines:** |residual| > 2 suggests significant cell contribution")
            
            # Visualization
            fig = self.viz_utils.create_independence_plot(
                contingency_table, expected, row_var, col_var
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _fishers_exact_test(self, confidence_level):
        """Fisher's exact test for 2x2 contingency tables"""
        st.subheader("Fisher's Exact Test")
        st.markdown("Exact test for independence in 2×2 contingency tables.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Input 2×2 Table")
            
            st.write("Enter cell values:")
            a = st.number_input("Cell (1,1):", min_value=0, value=10, key="fisher_a")
            b = st.number_input("Cell (1,2):", min_value=0, value=5, key="fisher_b")
            c = st.number_input("Cell (2,1):", min_value=0, value=3, key="fisher_c")
            d = st.number_input("Cell (2,2):", min_value=0, value=12, key="fisher_d")
            
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"],
                key="fisher_alt"
            )
        
        with col1:
            # Create contingency table
            table = np.array([[a, b], [c, d]])
            
            # Display table
            st.subheader("2×2 Contingency Table")
            
            table_df = pd.DataFrame(
                table,
                columns=['Column 1', 'Column 2'],
                index=['Row 1', 'Row 2']
            )
            
            # Add margins
            table_df['Row Total'] = table_df.sum(axis=1)
            col_totals = table_df.sum(axis=0)
            table_df.loc['Column Total'] = col_totals
            
            st.dataframe(table_df)
            
            # Check when to use Fisher's exact test
            st.subheader("Test Selection")
            
            # Expected frequencies for chi-square
            row_totals = table.sum(axis=1)
            col_totals = table.sum(axis=0)
            grand_total = table.sum()
            
            if grand_total == 0:
                st.error("Empty table - please enter data")
                return
            
            expected = np.outer(row_totals, col_totals) / grand_total
            min_expected = np.min(expected)
            
            if min_expected < 5 or grand_total < 30:
                st.success("✅ Fisher's exact test recommended")
                st.write(f"Minimum expected frequency: {min_expected:.2f}")
            else:
                st.info("ℹ️ Chi-square test also appropriate")
                st.write("Fisher's exact test provides exact p-values")
            
            # Perform Fisher's exact test
            st.subheader("Test Results")
            
            # SciPy Fisher's exact test
            if alternative == "two-sided":
                odds_ratio, p_value = stats.fisher_exact(table, alternative='two-sided')
            elif alternative == "greater":
                odds_ratio, p_value = stats.fisher_exact(table, alternative='greater')
            else:  # less
                odds_ratio, p_value = stats.fisher_exact(table, alternative='less')
            
            # Chi-square test for comparison
            chi2_stat, chi2_p, _, _ = stats.chi2_contingency(table)
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Fisher's p-value", f"{p_value:.6f}")
                st.metric("Chi-square p-value", f"{chi2_p:.6f}")
            
            with results_col2:
                st.metric("Odds Ratio", f"{odds_ratio:.4f}")
                st.metric("Sample Size", f"{grand_total}")
            
            with results_col3:
                alpha = 1 - confidence_level
                if p_value < alpha:
                    st.success("✅ Reject H₀")
                    st.write("Association exists")
                else:
                    st.info("❌ Fail to reject H₀")
                    st.write("No association")
            
            # Confidence interval for odds ratio
            st.subheader("Odds Ratio Analysis")
            
            if b > 0 and c > 0:
                # Exact confidence interval is complex, use approximation
                log_or = np.log(odds_ratio)
                se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
                z_critical = stats.norm.ppf(1 - (1-confidence_level)/2)
                
                ci_lower = np.exp(log_or - z_critical * se_log_or)
                ci_upper = np.exp(log_or + z_critical * se_log_or)
                
                or_col1, or_col2 = st.columns(2)
                
                with or_col1:
                    st.write(f"**{confidence_level*100:.0f}% CI for OR:**")
                    st.write(f"[{ci_lower:.4f}, {ci_upper:.4f}]")
                    
                    if 1 < ci_lower or 1 > ci_upper:
                        st.write("✅ OR significantly different from 1")
                    else:
                        st.write("❌ OR not significantly different from 1")
                
                with or_col2:
                    st.write("**OR Interpretation:**")
                    if odds_ratio > 1:
                        st.write(f"Row 1 has {odds_ratio:.2f}× odds of Column 1")
                    elif odds_ratio < 1:
                        st.write(f"Row 1 has {1/odds_ratio:.2f}× lower odds")
                    else:
                        st.write("Equal odds")
            else:
                st.warning("Cannot calculate OR confidence interval (zero cell)")
            
            # Additional measures for 2x2 table
            st.subheader("Additional Measures")
            
            measures_col1, measures_col2 = st.columns(2)
            
            with measures_col1:
                # Risk measures
                risk1 = a / (a + b) if (a + b) > 0 else 0
                risk2 = c / (c + d) if (c + d) > 0 else 0
                
                st.write("**Risk Analysis:**")
                st.write(f"Risk in Row 1: {risk1:.4f}")
                st.write(f"Risk in Row 2: {risk2:.4f}")
                
                if risk2 > 0:
                    relative_risk = risk1 / risk2
                    st.write(f"Relative Risk: {relative_risk:.4f}")
                
                risk_difference = risk1 - risk2
                st.write(f"Risk Difference: {risk_difference:.4f}")
            
            with measures_col2:
                # Sensitivity/Specificity (if applicable)
                st.write("**Diagnostic Measures (if applicable):**")
                
                # Assuming Row 1 = Disease+, Row 2 = Disease-
                # Column 1 = Test+, Column 2 = Test-
                sensitivity = a / (a + b) if (a + b) > 0 else 0
                specificity = d / (c + d) if (c + d) > 0 else 0
                
                st.write(f"Sensitivity: {sensitivity:.4f}")
                st.write(f"Specificity: {specificity:.4f}")
                
                ppv = a / (a + c) if (a + c) > 0 else 0
                npv = d / (b + d) if (b + d) > 0 else 0
                
                st.write(f"PPV: {ppv:.4f}")
                st.write(f"NPV: {npv:.4f}")
            
            # Exact vs approximate comparison
            st.subheader("Test Comparison")
            
            comparison_df = pd.DataFrame({
                'Test': ['Fisher\'s Exact', 'Chi-square', 'Difference'],
                'p-value': [f"{p_value:.6f}", f"{chi2_p:.6f}", f"{abs(p_value - chi2_p):.6f}"],
                'Comments': [
                    'Exact, no assumptions',
                    'Approximate, assumes large sample',
                    'Difference in p-values'
                ]
            })
            
            st.dataframe(comparison_df, hide_index=True)
            
            # Visualization
            fig = self.viz_utils.create_fisher_exact_plot(table, odds_ratio, p_value)
            st.plotly_chart(fig, use_container_width=True)
    
    def _mcnemars_test(self, confidence_level):
        """McNemar's test for paired binary data"""
        st.subheader("McNemar's Test")
        st.markdown("Tests for changes in paired binary responses (before/after, matched pairs).")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Input Paired Data")
            
            input_method = st.radio(
                "Input method:",
                ["2×2 Table", "Paired Observations"]
            )
            
            if input_method == "2×2 Table":
                st.write("Enter change table (Before vs After):")
                st.write("Rows: Before, Columns: After")
                
                # For McNemar's test: only off-diagonal cells matter
                both_positive = st.number_input("Both positive:", min_value=0, value=10)
                positive_to_negative = st.number_input("Positive → Negative:", min_value=0, value=3)
                negative_to_positive = st.number_input("Negative → Positive:", min_value=0, value=7)
                both_negative = st.number_input("Both negative:", min_value=0, value=20)
                
                mcnemar_table = np.array([
                    [both_positive, positive_to_negative],
                    [negative_to_positive, both_negative]
                ])
                
            else:  # Paired observations
                st.write("Enter paired binary data:")
                before_input = st.text_area(
                    "Before (1s and 0s):",
                    placeholder="1,0,1,1,0,1,0,0,1,1",
                    height=60
                )
                after_input = st.text_area(
                    "After (1s and 0s):",
                    placeholder="1,1,1,0,0,1,1,0,1,1",
                    height=60
                )
                
                if before_input and after_input:
                    try:
                        before_data = [int(x.strip()) for x in before_input.split(',')]
                        after_data = [int(x.strip()) for x in after_input.split(',')]
                        
                        if len(before_data) != len(after_data):
                            st.error("Before and after data must have same length")
                            return
                        
                        # Create McNemar table
                        both_positive = sum(1 for b, a in zip(before_data, after_data) if b == 1 and a == 1)
                        positive_to_negative = sum(1 for b, a in zip(before_data, after_data) if b == 1 and a == 0)
                        negative_to_positive = sum(1 for b, a in zip(before_data, after_data) if b == 0 and a == 1)
                        both_negative = sum(1 for b, a in zip(before_data, after_data) if b == 0 and a == 0)
                        
                        mcnemar_table = np.array([
                            [both_positive, positive_to_negative],
                            [negative_to_positive, both_negative]
                        ])
                        
                    except:
                        st.error("Invalid data format")
                        return
                else:
                    mcnemar_table = None
            
            if mcnemar_table is not None:
                use_continuity = st.checkbox("Use continuity correction", value=True)
        
        with col1:
            if mcnemar_table is None:
                st.warning("Please enter paired data")
                return
            
            # Display McNemar table
            st.subheader("McNemar's Table")
            
            table_df = pd.DataFrame(
                mcnemar_table,
                columns=['After: Positive', 'After: Negative'],
                index=['Before: Positive', 'Before: Negative']
            )
            
            # Add margins
            table_df['Row Total'] = table_df.sum(axis=1)
            col_totals = table_df.sum(axis=0)
            table_df.loc['Column Total'] = col_totals
            
            st.dataframe(table_df)
            
            # Extract key values
            b = positive_to_negative  # Changed from positive to negative
            c = negative_to_positive  # Changed from negative to positive
            
            # Check assumptions
            st.subheader("Assumption Checking")
            
            assumption_col1, assumption_col2 = st.columns(2)
            
            with assumption_col1:
                st.write("**Sample Size for Normal Approximation:**")
                if b + c >= 25:
                    st.success("✅ b + c ≥ 25 (normal approximation valid)")
                    use_normal = True
                elif b + c >= 10:
                    st.warning("⚠️ 10 ≤ b + c < 25 (marginal for normal approximation)")
                    use_normal = True
                else:
                    st.error("❌ b + c < 10 (use exact binomial test)")
                    use_normal = False
                
                st.write(f"Discordant pairs (b + c): {b + c}")
            
            with assumption_col2:
                st.write("**Paired Design:**")
                st.info("Assumes dependent observations (paired design)")
                st.write("**Binary Outcomes:**")
                st.info("Each observation has binary outcome")
            
            # Perform McNemar's test
            st.subheader("Test Results")
            
            if use_normal:
                # McNemar's chi-square test
                if use_continuity:
                    mcnemar_stat = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
                else:
                    mcnemar_stat = (b - c)**2 / (b + c) if (b + c) > 0 else 0
                
                p_value_mcnemar = 1 - stats.chi2.cdf(mcnemar_stat, 1)
                
                # Exact binomial test (for comparison)
                if b + c > 0:
                    p_value_exact = 2 * min(
                        stats.binom.cdf(min(b, c), b + c, 0.5),
                        1 - stats.binom.cdf(min(b, c) - 1, b + c, 0.5)
                    )
                else:
                    p_value_exact = 1.0
            else:
                # Use exact test only
                mcnemar_stat = None
                if b + c > 0:
                    p_value_exact = 2 * min(
                        stats.binom.cdf(min(b, c), b + c, 0.5),
                        1 - stats.binom.cdf(min(b, c) - 1, b + c, 0.5)
                    )
                else:
                    p_value_exact = 1.0
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                if use_normal:
                    st.metric("McNemar χ²", f"{mcnemar_stat:.4f}")
                    st.metric("p-value (χ²)", f"{p_value_mcnemar:.6f}")
                else:
                    st.info("Using exact test only")
                
                st.metric("p-value (exact)", f"{p_value_exact:.6f}")
            
            with results_col2:
                st.metric("Pos → Neg (b)", b)
                st.metric("Neg → Pos (c)", c)
                st.metric("Discordant pairs", b + c)
            
            with results_col3:
                alpha = 1 - confidence_level
                p_value_to_use = p_value_mcnemar if use_normal else p_value_exact
                
                if p_value_to_use < alpha:
                    st.success("✅ Reject H₀")
                    st.write("Significant change")
                else:
                    st.info("❌ Fail to reject H₀")
                    st.write("No significant change")
            
            # Effect size and additional measures
            st.subheader("Effect Size and Change Analysis")
            
            effect_col1, effect_col2 = st.columns(2)
            
            with effect_col1:
                # Proportion of discordant pairs
                total_pairs = mcnemar_table.sum()
                prop_discordant = (b + c) / total_pairs if total_pairs > 0 else 0
                
                st.metric("Proportion discordant", f"{prop_discordant:.4f}")
                
                # Odds ratio for change
                if c > 0:
                    odds_ratio_change = b / c
                    st.metric("Odds ratio (b/c)", f"{odds_ratio_change:.4f}")
                    
                    if odds_ratio_change > 1:
                        st.write("More positive → negative changes")
                    elif odds_ratio_change < 1:
                        st.write("More negative → positive changes")
                    else:
                        st.write("Equal changes both directions")
                else:
                    st.info("Cannot calculate OR (c = 0)")
            
            with effect_col2:
                # Marginal homogeneity
                before_positive = both_positive + positive_to_negative
                after_positive = both_positive + negative_to_positive
                
                st.write("**Marginal Frequencies:**")
                st.write(f"Before positive: {before_positive}")
                st.write(f"After positive: {after_positive}")
                st.write(f"Net change: {after_positive - before_positive}")
                
                # Proportion change
                if before_positive > 0:
                    prop_change = (after_positive - before_positive) / before_positive
                    st.metric("Relative change", f"{prop_change:.4f}")
            
            # Confidence interval for proportion difference
            st.subheader("Confidence Intervals")
            
            # Proportion before and after
            prop_before = before_positive / total_pairs if total_pairs > 0 else 0
            prop_after = after_positive / total_pairs if total_pairs > 0 else 0
            prop_diff = prop_after - prop_before
            
            # Standard error for paired proportions
            se_diff = np.sqrt((b + c) / total_pairs**2) if total_pairs > 0 else 0
            z_critical = stats.norm.ppf(1 - (1-confidence_level)/2)
            
            ci_lower = prop_diff - z_critical * se_diff
            ci_upper = prop_diff + z_critical * se_diff
            
            ci_col1, ci_col2 = st.columns(2)
            
            with ci_col1:
                st.write(f"**CI for Proportion Difference:**")
                st.write(f"[{ci_lower:.4f}, {ci_upper:.4f}]")
                
                if 0 < ci_lower or 0 > ci_upper:
                    st.write("✅ Zero not in CI (significant change)")
                else:
                    st.write("❌ Zero in CI (no significant change)")
            
            with ci_col2:
                st.write("**Individual Proportions:**")
                st.write(f"Before: {prop_before:.4f}")
                st.write(f"After: {prop_after:.4f}")
                st.write(f"Difference: {prop_diff:.4f}")
            
            # Visualization
            fig = self.viz_utils.create_mcnemar_plot(
                mcnemar_table, before_positive, after_positive
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _cochrans_q_test(self, confidence_level):
        """Cochran's Q test for multiple related binary variables"""
        st.subheader("Cochran's Q Test")
        st.markdown("Tests for differences in proportions across multiple related binary variables.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Input Data")
            
            input_method = st.radio(
                "Input method:",
                ["Matrix Entry", "Upload Data"]
            )
            
            if input_method == "Matrix Entry":
                n_subjects = st.number_input("Number of subjects:", min_value=3, max_value=100, value=10)
                n_treatments = st.number_input("Number of treatments/conditions:", min_value=3, max_value=10, value=3)
                
                st.write("Enter binary responses (1 = success, 0 = failure):")
                
                # Create data matrix
                data_matrix = []
                for i in range(n_subjects):
                    subject_row = []
                    st.write(f"Subject {i+1}:")
                    cols = st.columns(n_treatments)
                    for j in range(n_treatments):
                        with cols[j]:
                            response = st.selectbox(
                                f"T{j+1}",
                                [0, 1],
                                key=f"subj_{i}_treat_{j}"
                            )
                            subject_row.append(response)
                    data_matrix.append(subject_row)
                
                data_matrix = np.array(data_matrix)
                treatment_names = [f"Treatment {j+1}" for j in range(n_treatments)]
                
            else:  # Upload Data
                st.info("Upload CSV with subjects as rows, treatments as columns")
                st.warning("Matrix entry required for this implementation")
                return
        
        with col1:
            if len(data_matrix) == 0:
                st.warning("Please enter response data")
                return
            
            # Data summary
            st.subheader("Data Summary")
            
            # Display data matrix
            data_df = pd.DataFrame(data_matrix, columns=treatment_names)
            data_df.index = [f"Subject {i+1}" for i in range(n_subjects)]
            st.dataframe(data_df)
            
            # Calculate summary statistics
            treatment_sums = data_matrix.sum(axis=0)  # Successes per treatment
            subject_sums = data_matrix.sum(axis=1)    # Successes per subject
            grand_total = data_matrix.sum()
            
            summary_df = pd.DataFrame({
                'Treatment': treatment_names,
                'Successes': treatment_sums,
                'Failures': n_subjects - treatment_sums,
                'Proportion': treatment_sums / n_subjects
            })
            
            st.dataframe(summary_df, hide_index=True)
            
            # Check assumptions
            st.subheader("Assumption Checking")
            
            assumption_col1, assumption_col2 = st.columns(2)
            
            with assumption_col1:
                st.write("**Sample Size:**")
                if n_subjects >= 10:
                    st.success("✅ Adequate number of subjects")
                else:
                    st.warning("⚠️ Small number of subjects")
                
                st.write(f"Subjects: {n_subjects}")
                st.write(f"Treatments: {n_treatments}")
            
            with assumption_col2:
                st.write("**Related Measurements:**")
                st.info("Assumes repeated measures on same subjects")
                
                # Check for subjects with all 0s or all 1s
                all_zeros = np.sum(subject_sums == 0)
                all_ones = np.sum(subject_sums == n_treatments)
                
                st.write(f"Subjects with all 0s: {all_zeros}")
                st.write(f"Subjects with all 1s: {all_ones}")
                
                if all_zeros + all_ones > 0.5 * n_subjects:
                    st.warning("⚠️ Many subjects with uniform responses")
            
            # Cochran's Q Test
            st.subheader("Test Results")
            
            # Calculate Q statistic
            k = n_treatments
            n = n_subjects
            
            # Q = (k-1) * [k * sum(Tj^2) - (sum(Tj))^2] / [k * sum(Li) - sum(Li^2)]
            sum_tj_squared = np.sum(treatment_sums**2)
            sum_tj_total_squared = (np.sum(treatment_sums))**2
            sum_li = np.sum(subject_sums)
            sum_li_squared = np.sum(subject_sums**2)
            
            denominator = k * sum_li - sum_li_squared
            
            if denominator == 0:
                st.error("Cannot compute Q statistic (denominator = 0)")
                st.write("This occurs when all subjects have identical response patterns")
                return
            
            q_statistic = (k - 1) * (k * sum_tj_squared - sum_tj_total_squared) / denominator
            
            # Degrees of freedom and p-value
            df = k - 1
            p_value = 1 - stats.chi2.cdf(q_statistic, df)
            
            # Effect size (Kendall's W for binary data)
            # W = Q / (N * (k-1))
            kendalls_w = q_statistic / (n * (k - 1))
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Q statistic", f"{q_statistic:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Degrees of freedom", df)
                st.metric("Kendall's W", f"{kendalls_w:.4f}")
            
            with results_col3:
                alpha = 1 - confidence_level
                critical_value = stats.chi2.ppf(confidence_level, df)
                st.metric("Critical value", f"{critical_value:.4f}")
                
                if p_value < alpha:
                    st.success("✅ Reject H₀")
                    st.write("Treatments differ")
                else:
                    st.info("❌ Fail to reject H₀")
                    st.write("No difference")
            
            # Effect size interpretation
            st.subheader("Effect Size Interpretation")
            
            if kendalls_w < 0.1:
                effect_magnitude = "negligible"
            elif kendalls_w < 0.3:
                effect_magnitude = "small"
            elif kendalls_w < 0.5:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            
            st.write(f"**Kendall's W magnitude:** {effect_magnitude}")
            st.write(f"**Interpretation:** {kendalls_w:.3f} indicates {'strong' if kendalls_w > 0.7 else 'moderate' if kendalls_w > 0.3 else 'weak'} agreement in treatment rankings")
            
            # Post-hoc analysis
            if p_value < alpha and n_treatments > 2:
                st.subheader("Post-hoc Pairwise Comparisons")
                st.info("Performing pairwise McNemar tests with Bonferroni correction")
                
                n_comparisons = n_treatments * (n_treatments - 1) // 2
                bonferroni_alpha = alpha / n_comparisons
                
                pairwise_results = []
                
                for i in range(n_treatments):
                    for j in range(i + 1, n_treatments):
                        # Extract paired data for treatments i and j
                        treat_i = data_matrix[:, i]
                        treat_j = data_matrix[:, j]
                        
                        # Create McNemar table
                        both_success = np.sum((treat_i == 1) & (treat_j == 1))
                        i_success_j_fail = np.sum((treat_i == 1) & (treat_j == 0))
                        i_fail_j_success = np.sum((treat_i == 0) & (treat_j == 1))
                        both_fail = np.sum((treat_i == 0) & (treat_j == 0))
                        
                        # McNemar test (exact)
                        discordant = i_success_j_fail + i_fail_j_success
                        if discordant > 0:
                            p_mcnemar = 2 * min(
                                stats.binom.cdf(min(i_success_j_fail, i_fail_j_success), discordant, 0.5),
                                1 - stats.binom.cdf(min(i_success_j_fail, i_fail_j_success) - 1, discordant, 0.5)
                            )
                        else:
                            p_mcnemar = 1.0
                        
                        significant = p_mcnemar < bonferroni_alpha
                        
                        pairwise_results.append({
                            'Comparison': f"{treatment_names[i]} vs {treatment_names[j]}",
                            'Discordant pairs': discordant,
                            'p-value': f"{p_mcnemar:.6f}",
                            'Significant': "✅" if significant else "❌"
                        })
                
                st.dataframe(pd.DataFrame(pairwise_results), hide_index=True)
                st.write(f"**Bonferroni-corrected α:** {bonferroni_alpha:.6f}")
            
            # Detailed statistics table
            st.subheader("Detailed Statistics")
            
            detailed_stats = []
            for j, name in enumerate(treatment_names):
                prop = treatment_sums[j] / n_subjects
                
                # Wilson confidence interval for proportion
                z_crit = stats.norm.ppf(1 - (1-confidence_level)/2)
                wilson_center = (prop + z_crit**2 / (2 * n_subjects)) / (1 + z_crit**2 / n_subjects)
                wilson_margin = z_crit * np.sqrt(prop * (1 - prop) / n_subjects + z_crit**2 / (4 * n_subjects**2)) / (1 + z_crit**2 / n_subjects)
                
                detailed_stats.append({
                    'Treatment': name,
                    'Successes': treatment_sums[j],
                    'Proportion': f"{prop:.4f}",
                    'CI Lower': f"{wilson_center - wilson_margin:.4f}",
                    'CI Upper': f"{wilson_center + wilson_margin:.4f}"
                })
            
            st.dataframe(pd.DataFrame(detailed_stats), hide_index=True)
            
            # Visualization
            fig = self.viz_utils.create_cochrans_q_plot(
                treatment_names, treatment_sums, n_subjects
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _bootstrap_proportion_tests(self, confidence_level, bootstrap_iterations):
        """Bootstrap methods for proportion tests"""
        st.subheader("Bootstrap Proportion Tests")
        st.markdown("Robust proportion testing using bootstrap resampling methods.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Test Configuration")
            
            test_type = st.selectbox(
                "Bootstrap test type:",
                ["One proportion", "Two proportions", "Multiple proportions"]
            )
            
            if test_type == "One proportion":
                st.write("**Sample Data:**")
                successes = st.number_input("Successes:", min_value=0, value=25)
                n_trials = st.number_input("Total trials:", min_value=1, value=100)
                null_prop = st.number_input("Null proportion:", min_value=0.0, max_value=1.0, value=0.2)
                
            elif test_type == "Two proportions":
                st.write("**Group 1:**")
                x1 = st.number_input("Successes Group 1:", min_value=0, value=30)
                n1 = st.number_input("Total Group 1:", min_value=1, value=150)
                
                st.write("**Group 2:**")
                x2 = st.number_input("Successes Group 2:", min_value=0, value=20)
                n2 = st.number_input("Total Group 2:", min_value=1, value=120)
                
            else:  # Multiple proportions
                st.write("**Multiple Groups:**")
                n_groups = st.number_input("Number of groups:", min_value=3, max_value=8, value=3)
                
                group_data = []
                for i in range(n_groups):
                    st.write(f"Group {i+1}:")
                    cols = st.columns(2)
                    with cols[0]:
                        successes = st.number_input(f"Successes {i+1}:", min_value=0, value=20, key=f"boot_succ_{i}")
                    with cols[1]:
                        total = st.number_input(f"Total {i+1}:", min_value=1, value=100, key=f"boot_total_{i}")
                    group_data.append((successes, total))
            
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"],
                key="bootstrap_prop_alt"
            )
        
        with col1:
            # Perform bootstrap test based on type
            if test_type == "One proportion":
                self._bootstrap_one_proportion(
                    successes, n_trials, null_prop, alternative, 
                    confidence_level, bootstrap_iterations
                )
            elif test_type == "Two proportions":
                self._bootstrap_two_proportions(
                    x1, n1, x2, n2, alternative,
                    confidence_level, bootstrap_iterations
                )
            else:  # Multiple proportions
                self._bootstrap_multiple_proportions(
                    group_data, confidence_level, bootstrap_iterations
                )
    
    def _bootstrap_one_proportion(self, successes, n_trials, null_prop, alternative, confidence_level, bootstrap_iterations):
        """Bootstrap test for one proportion"""
        st.subheader("Bootstrap One-Proportion Test")
        
        observed_prop = successes / n_trials
        
        # Data summary
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Observed Proportion", f"{observed_prop:.4f}")
            st.metric("Sample Size", n_trials)
        
        with summary_col2:
            st.metric("Null Proportion", f"{null_prop:.4f}")
            st.metric("Successes", successes)
        
        with summary_col3:
            st.metric("Difference", f"{observed_prop - null_prop:.4f}")
            st.metric("Failures", n_trials - successes)
        
        # Bootstrap test
        with st.spinner(f"Performing {bootstrap_iterations:,} bootstrap iterations..."):
            # Generate data under null hypothesis
            null_successes = np.random.binomial(n_trials, null_prop, bootstrap_iterations)
            bootstrap_props = null_successes / n_trials
            
            # Test statistic: difference from null
            observed_diff = observed_prop - null_prop
            bootstrap_diffs = bootstrap_props - null_prop
        
        # Calculate p-value
        if alternative == "two-sided":
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        elif alternative == "greater":
            p_value = np.mean(bootstrap_diffs >= observed_diff)
        else:  # less
            p_value = np.mean(bootstrap_diffs <= observed_diff)
        
        # Bootstrap confidence interval for proportion
        # Use percentile method
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_props, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_props, 100 * (1 - alpha/2))
        
        # Display results
        st.subheader("Bootstrap Test Results")
        
        results_col1, results_col2, results_col3 = st.columns(3)
        
        with results_col1:
            st.metric("Bootstrap p-value", f"{p_value:.6f}")
            st.metric("Bootstrap iterations", f"{bootstrap_iterations:,}")
        
        with results_col2:
            st.metric("CI Lower", f"{ci_lower:.4f}")
            st.metric("CI Upper", f"{ci_upper:.4f}")
        
        with results_col3:
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success("✅ Reject H₀")
            else:
                st.info("❌ Fail to reject H₀")
        
        # Visualization
        fig = go.Figure()
        
        # Bootstrap distribution
        fig.add_trace(go.Histogram(
            x=bootstrap_props,
            nbinsx=50,
            name="Bootstrap Distribution",
            opacity=0.7
        ))
        
        # Observed proportion
        fig.add_vline(
            x=observed_prop,
            line_dash="dash",
            line_color="red",
            annotation_text="Observed Proportion"
        )
        
        # Null proportion
        fig.add_vline(
            x=null_prop,
            line_dash="dot",
            line_color="blue",
            annotation_text="Null Proportion"
        )
        
        # Confidence interval
        fig.add_vline(x=ci_lower, line_dash="dot", line_color="green")
        fig.add_vline(x=ci_upper, line_dash="dot", line_color="green")
        
        fig.update_layout(
            title="Bootstrap Distribution of Proportion",
            xaxis_title="Proportion",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _bootstrap_two_proportions(self, x1, n1, x2, n2, alternative, confidence_level, bootstrap_iterations):
        """Bootstrap test for two proportions"""
        st.subheader("Bootstrap Two-Proportion Test")
        
        p1_observed = x1 / n1
        p2_observed = x2 / n2
        diff_observed = p1_observed - p2_observed
        
        # Data summary
        summary_data = pd.DataFrame({
            'Group': ['Group 1', 'Group 2'],
            'Successes': [x1, x2],
            'Total': [n1, n2],
            'Proportion': [f"{p1_observed:.4f}", f"{p2_observed:.4f}"],
            'Failures': [n1-x1, n2-x2]
        })
        
        st.dataframe(summary_data, hide_index=True)
        
        st.metric("Observed Difference (p₁ - p₂)", f"{diff_observed:.4f}")
        
        # Bootstrap test under null hypothesis (no difference)
        with st.spinner(f"Performing {bootstrap_iterations:,} bootstrap iterations..."):
            # Pool data under null hypothesis
            pooled_prop = (x1 + x2) / (n1 + n2)
            
            # Bootstrap samples under null
            bootstrap_x1 = np.random.binomial(n1, pooled_prop, bootstrap_iterations)
            bootstrap_x2 = np.random.binomial(n2, pooled_prop, bootstrap_iterations)
            
            bootstrap_p1 = bootstrap_x1 / n1
            bootstrap_p2 = bootstrap_x2 / n2
            bootstrap_diffs = bootstrap_p1 - bootstrap_p2
        
        # Calculate p-value
        if alternative == "two-sided":
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(diff_observed))
        elif alternative == "greater":
            p_value = np.mean(bootstrap_diffs >= diff_observed)
        else:  # less
            p_value = np.mean(bootstrap_diffs <= diff_observed)
        
        # Confidence interval for difference
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha/2))
        
        # Display results
        st.subheader("Bootstrap Test Results")
        
        results_col1, results_col2, results_col3 = st.columns(3)
        
        with results_col1:
            st.metric("Bootstrap p-value", f"{p_value:.6f}")
            st.metric("Pooled proportion", f"{pooled_prop:.4f}")
        
        with results_col2:
            st.metric("CI Lower", f"{ci_lower:.4f}")
            st.metric("CI Upper", f"{ci_upper:.4f}")
        
        with results_col3:
            alpha_level = 1 - confidence_level
            if p_value < alpha_level:
                st.success("✅ Reject H₀")
            else:
                st.info("❌ Fail to reject H₀")
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=bootstrap_diffs,
            nbinsx=50,
            name="Bootstrap Differences",
            opacity=0.7
        ))
        
        fig.add_vline(
            x=diff_observed,
            line_dash="dash",
            line_color="red",
            annotation_text="Observed Difference"
        )
        
        fig.add_vline(x=0, line_dash="dot", line_color="blue", annotation_text="Null (No Difference)")
        fig.add_vline(x=ci_lower, line_dash="dot", line_color="green")
        fig.add_vline(x=ci_upper, line_dash="dot", line_color="green")
        
        fig.update_layout(
            title="Bootstrap Distribution of Proportion Difference",
            xaxis_title="Difference in Proportions",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _bootstrap_multiple_proportions(self, group_data, confidence_level, bootstrap_iterations):
        """Bootstrap test for multiple proportions"""
        st.subheader("Bootstrap Multiple Proportion Test")
        
        # Extract data
        successes = [x for x, n in group_data]
        totals = [n for x, n in group_data]
        proportions = [x/n for x, n in group_data]
        
        # Data summary
        summary_data = pd.DataFrame({
            'Group': [f"Group {i+1}" for i in range(len(group_data))],
            'Successes': successes,
            'Total': totals,
            'Proportion': [f"{p:.4f}" for p in proportions],
            'Failures': [n-x for x, n in group_data]
        })
        
        st.dataframe(summary_data, hide_index=True)
        
        # Test statistic: chi-square-like statistic
        grand_total = sum(totals)
        grand_successes = sum(successes)
        pooled_prop = grand_successes / grand_total
        
        # Observed test statistic
        observed_stat = sum((x - n * pooled_prop)**2 / (n * pooled_prop * (1 - pooled_prop)) 
                          for x, n in group_data)
        
        # Bootstrap test
        with st.spinner(f"Performing {bootstrap_iterations:,} bootstrap iterations..."):
            bootstrap_stats = []
            
            for _ in range(bootstrap_iterations):
                # Generate bootstrap samples under null (all groups have same proportion)
                bootstrap_successes = [np.random.binomial(n, pooled_prop) for n in totals]
                
                # Calculate test statistic
                boot_stat = sum((x - n * pooled_prop)**2 / (n * pooled_prop * (1 - pooled_prop)) 
                               for x, n in zip(bootstrap_successes, totals))
                bootstrap_stats.append(boot_stat)
            
            bootstrap_stats = np.array(bootstrap_stats)
        
        # P-value (one-tailed, since test statistic is always positive)
        p_value = np.mean(bootstrap_stats >= observed_stat)
        
        # Display results
        st.subheader("Bootstrap Test Results")
        
        results_col1, results_col2, results_col3 = st.columns(3)
        
        with results_col1:
            st.metric("Test Statistic", f"{observed_stat:.4f}")
            st.metric("Bootstrap p-value", f"{p_value:.6f}")
        
        with results_col2:
            st.metric("Pooled Proportion", f"{pooled_prop:.4f}")
            st.metric("Bootstrap iterations", f"{bootstrap_iterations:,}")
        
        with results_col3:
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success("✅ Reject H₀")
                st.write("Proportions differ")
            else:
                st.info("❌ Fail to reject H₀")
                st.write("Proportions equal")
        
        # Individual proportion confidence intervals
        st.subheader("Individual Proportion Confidence Intervals")
        
        ci_data = []
        alpha = 1 - confidence_level
        
        for i, (x, n) in enumerate(group_data):
            # Bootstrap CI for individual proportion
            bootstrap_props = np.random.binomial(n, x/n, bootstrap_iterations) / n
            ci_lower = np.percentile(bootstrap_props, 100 * alpha/2)
            ci_upper = np.percentile(bootstrap_props, 100 * (1 - alpha/2))
            
            ci_data.append({
                'Group': f"Group {i+1}",
                'Proportion': f"{x/n:.4f}",
                'CI Lower': f"{ci_lower:.4f}",
                'CI Upper': f"{ci_upper:.4f}",
                'Contains Pooled': "Yes" if ci_lower <= pooled_prop <= ci_upper else "No"
            })
        
        st.dataframe(pd.DataFrame(ci_data), hide_index=True)
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=bootstrap_stats,
            nbinsx=50,
            name="Bootstrap Test Statistics",
            opacity=0.7
        ))
        
        fig.add_vline(
            x=observed_stat,
            line_dash="dash",
            line_color="red",
            annotation_text="Observed Statistic"
        )
        
        fig.update_layout(
            title="Bootstrap Distribution of Test Statistic",
            xaxis_title="Test Statistic",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
