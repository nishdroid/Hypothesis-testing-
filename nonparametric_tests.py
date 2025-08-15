import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from .statistical_utils import StatisticalUtils
from .visualization_utils import VisualizationUtils

class NonParametricTests:
    def __init__(self):
        self.stat_utils = StatisticalUtils()
        self.viz_utils = VisualizationUtils()
    
    def render(self, data, confidence_level, bootstrap_iterations):
        st.header("📉 Non-parametric Statistical Tests")
        st.markdown("Distribution-free tests that make minimal assumptions about data distribution.")
        
        # Test selection
        test_type = st.selectbox(
            "Select Non-parametric Test:",
            [
                "Wilcoxon signed-rank test",
                "Mann-Whitney U test",
                "Kruskal-Wallis test",
                "Friedman test",
                "Wilcoxon rank-sum test",
                "Sign test",
                "Runs test for randomness",
                "Kolmogorov-Smirnov two-sample test"
            ]
        )
        
        if test_type == "Wilcoxon signed-rank test":
            self._wilcoxon_signed_rank(data, confidence_level)
        elif test_type == "Mann-Whitney U test":
            self._mann_whitney_u(data, confidence_level)
        elif test_type == "Kruskal-Wallis test":
            self._kruskal_wallis(data, confidence_level)
        elif test_type == "Friedman test":
            self._friedman_test(data, confidence_level)
        elif test_type == "Wilcoxon rank-sum test":
            self._wilcoxon_rank_sum(data, confidence_level)
        elif test_type == "Sign test":
            self._sign_test(data, confidence_level)
        elif test_type == "Runs test for randomness":
            self._runs_test(data, confidence_level)
        elif test_type == "Kolmogorov-Smirnov two-sample test":
            self._ks_two_sample(data, confidence_level)
    
    def _wilcoxon_signed_rank(self, data, confidence_level):
        """Wilcoxon signed-rank test implementation"""
        st.subheader("Wilcoxon Signed-Rank Test")
        st.markdown("Tests whether the median of differences differs from zero (paired data).")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if data is not None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    var1 = st.selectbox("Variable 1 (before):", numeric_cols)
                    var2 = st.selectbox("Variable 2 (after):", numeric_cols, index=1)
                    use_data = True
                else:
                    st.warning("Need at least 2 numeric variables")
                    use_data = False
            else:
                use_data = False
            
            if not use_data:
                st.subheader("Enter Raw Differences")
                diff_input = st.text_area(
                    "Enter differences (comma-separated):",
                    placeholder="1.2, -0.5, 2.1, -1.8, 0.3"
                )
            
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"],
                key="wilcoxon_alt"
            )
        
        with col1:
            if use_data and data is not None and var1 != var2:
                # Calculate differences
                clean_data = data[[var1, var2]].dropna()
                differences = clean_data[var2] - clean_data[var1]
                
                # Remove zeros (ties at zero)
                non_zero_diff = differences[differences != 0]
                
                if len(non_zero_diff) < 6:
                    st.warning("Small sample size - results may be unreliable")
                
            elif not use_data and diff_input:
                try:
                    differences = pd.Series([float(x.strip()) for x in diff_input.split(',')])
                    non_zero_diff = differences[differences != 0]
                    clean_data = None
                except:
                    st.error("Invalid input format")
                    return
            else:
                st.warning("Please provide data")
                return
            
            if len(non_zero_diff) == 0:
                st.error("No non-zero differences found")
                return
            
            # Perform Wilcoxon signed-rank test
            try:
                statistic, p_value = stats.wilcoxon(non_zero_diff, alternative=alternative)
            except ValueError as e:
                st.error(f"Error performing test: {str(e)}")
                return
            
            # Calculate effect size (r = Z/sqrt(N))
            n = len(non_zero_diff)
            z_score = stats.norm.ppf(1 - p_value/2) if alternative == "two-sided" else stats.norm.ppf(1 - p_value)
            effect_size_r = abs(z_score) / np.sqrt(n)
            
            # Display results
            st.subheader("Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("W-statistic", f"{statistic:.0f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Effect size (r)", f"{effect_size_r:.4f}")
                st.metric("Sample size", f"{n}")
            
            with results_col3:
                median_diff = non_zero_diff.median()
                st.metric("Median difference", f"{median_diff:.4f}")
                positive_ranks = sum(non_zero_diff > 0)
                st.metric("Positive ranks", f"{positive_ranks}")
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write("There is a significant difference in paired measurements")
            else:
                st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write("No significant difference in paired measurements")
            
            # Effect size interpretation
            if effect_size_r < 0.1:
                effect_magnitude = "negligible"
            elif effect_size_r < 0.3:
                effect_magnitude = "small"
            elif effect_size_r < 0.5:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            
            st.write(f"**Effect size magnitude:** {effect_magnitude}")
            
            # Visualization
            if clean_data is not None:
                fig = self.viz_utils.create_paired_nonparametric_plot(
                    clean_data[var1], clean_data[var2], differences, var1, var2
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=differences, name="Differences", nbinsx=20))
                fig.update_layout(title="Distribution of Differences")
                st.plotly_chart(fig, use_container_width=True)
    
    def _mann_whitney_u(self, data, confidence_level):
        """Mann-Whitney U test implementation"""
        st.subheader("Mann-Whitney U Test")
        st.markdown("Tests whether two independent samples come from the same distribution.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if data is not None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols and categorical_cols:
                    dependent_var = st.selectbox("Dependent variable:", numeric_cols)
                    grouping_var = st.selectbox("Grouping variable:", categorical_cols)
                    use_data = True
                else:
                    st.warning("Need both numeric and categorical variables")
                    use_data = False
            else:
                use_data = False
            
            if not use_data:
                st.subheader("Enter Raw Data")
                group1_input = st.text_area("Group 1 data:", placeholder="1.2, 2.3, 3.4")
                group2_input = st.text_area("Group 2 data:", placeholder="2.1, 3.2, 4.3")
            
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"],
                key="mannwhitney_alt"
            )
        
        with col1:
            if use_data and data is not None:
                # Get unique groups
                unique_groups = data[grouping_var].dropna().unique()
                if len(unique_groups) != 2:
                    st.error("Grouping variable must have exactly 2 categories")
                    return
                
                group1_data = data[data[grouping_var] == unique_groups[0]][dependent_var].dropna()
                group2_data = data[data[grouping_var] == unique_groups[1]][dependent_var].dropna()
                group1_name, group2_name = str(unique_groups[0]), str(unique_groups[1])
                
            elif not use_data and group1_input and group2_input:
                try:
                    group1_data = pd.Series([float(x.strip()) for x in group1_input.split(',')])
                    group2_data = pd.Series([float(x.strip()) for x in group2_input.split(',')])
                    group1_name, group2_name = "Group 1", "Group 2"
                except:
                    st.error("Invalid input format")
                    return
            else:
                st.warning("Please provide data for both groups")
                return
            
            if len(group1_data) == 0 or len(group2_data) == 0:
                st.error("Both groups must have data")
                return
            
            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                group1_data, group2_data, alternative=alternative
            )
            
            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(group1_data), len(group2_data)
            u1 = statistic
            u2 = n1 * n2 - u1
            effect_size = 1 - (2 * min(u1, u2)) / (n1 * n2)
            
            # Display results
            st.subheader("Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("U-statistic", f"{statistic:.0f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Effect size (r)", f"{effect_size:.4f}")
                st.metric("Total N", f"{n1 + n2}")
            
            with results_col3:
                st.metric(f"{group1_name} median", f"{group1_data.median():.4f}")
                st.metric(f"{group2_name} median", f"{group2_data.median():.4f}")
            
            # Group statistics
            st.subheader("Group Statistics")
            stats_df = pd.DataFrame({
                'Group': [group1_name, group2_name],
                'N': [n1, n2],
                'Median': [f"{group1_data.median():.4f}", f"{group2_data.median():.4f}"],
                'Mean Rank': [
                    f"{stats.rankdata(np.concatenate([group1_data, group2_data]))[:n1].mean():.2f}",
                    f"{stats.rankdata(np.concatenate([group1_data, group2_data]))[n1:].mean():.2f}"
                ],
                'Min': [f"{group1_data.min():.4f}", f"{group2_data.min():.4f}"],
                'Max': [f"{group1_data.max():.4f}", f"{group2_data.max():.4f}"]
            })
            st.dataframe(stats_df, hide_index=True)
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write("The distributions of the two groups are significantly different")
            else:
                st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write("No significant difference between group distributions")
            
            # Effect size interpretation
            if effect_size < 0.1:
                effect_magnitude = "negligible"
            elif effect_size < 0.3:
                effect_magnitude = "small"
            elif effect_size < 0.5:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            
            st.write(f"**Effect size magnitude:** {effect_magnitude}")
            
            # Visualization
            fig = self.viz_utils.create_nonparametric_comparison_plot(
                group1_data, group2_data, group1_name, group2_name, 
                dependent_var if use_data else "Value"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _kruskal_wallis(self, data, confidence_level):
        """Kruskal-Wallis test implementation"""
        st.subheader("Kruskal-Wallis Test")
        st.markdown("Non-parametric alternative to one-way ANOVA for multiple independent groups.")
        
        if data is None:
            st.warning("Please upload data to perform Kruskal-Wallis test")
            return
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_cols or not categorical_cols:
            st.error("Need both numeric and categorical variables")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            dependent_var = st.selectbox("Dependent variable:", numeric_cols)
            factor_var = st.selectbox("Factor variable:", categorical_cols)
        
        with col1:
            # Get groups
            groups = []
            group_names = []
            for group_name in data[factor_var].unique():
                if pd.notna(group_name):
                    group_data = data[data[factor_var] == group_name][dependent_var].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                        group_names.append(str(group_name))
            
            if len(groups) < 3:
                st.warning("Kruskal-Wallis test is typically used for 3+ groups. Consider Mann-Whitney U for 2 groups.")
            
            # Perform Kruskal-Wallis test
            h_statistic, p_value = stats.kruskal(*groups)
            
            # Calculate effect size (epsilon-squared)
            total_n = sum(len(group) for group in groups)
            k = len(groups)
            epsilon_squared = (h_statistic - k + 1) / (total_n - k)
            
            # Display results
            st.subheader("Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("H-statistic", f"{h_statistic:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("ε² (epsilon-squared)", f"{epsilon_squared:.4f}")
                st.metric("Number of groups", len(groups))
            
            with results_col3:
                st.metric("Total observations", total_n)
                st.metric("df", k - 1)
            
            # Group statistics with ranks
            st.subheader("Group Statistics")
            
            # Calculate overall ranks
            all_data = np.concatenate(groups)
            overall_ranks = stats.rankdata(all_data)
            
            rank_start = 0
            group_stats = []
            for i, (group, name) in enumerate(zip(groups, group_names)):
                group_size = len(group)
                group_ranks = overall_ranks[rank_start:rank_start + group_size]
                rank_start += group_size
                
                group_stats.append({
                    'Group': name,
                    'N': group_size,
                    'Median': f"{group.median():.4f}",
                    'Mean Rank': f"{group_ranks.mean():.2f}",
                    'Sum of Ranks': f"{group_ranks.sum():.0f}",
                    'Min': f"{group.min():.4f}",
                    'Max': f"{group.max():.4f}"
                })
            
            st.dataframe(pd.DataFrame(group_stats), hide_index=True)
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write("There is a significant difference between at least two groups")
                st.info("Consider post-hoc pairwise comparisons to identify which groups differ")
            else:
                st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write("No significant differences between groups")
            
            # Effect size interpretation
            if epsilon_squared < 0.01:
                effect_magnitude = "negligible"
            elif epsilon_squared < 0.06:
                effect_magnitude = "small"
            elif epsilon_squared < 0.14:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            
            st.write(f"**Effect size magnitude:** {effect_magnitude}")
            
            # Visualization
            fig = self.viz_utils.create_kruskal_wallis_plot(groups, group_names, dependent_var)
            st.plotly_chart(fig, use_container_width=True)
            
            # Post-hoc analysis if significant
            if p_value < alpha and len(groups) > 2:
                st.subheader("Post-hoc Pairwise Comparisons")
                st.info("Performing pairwise Mann-Whitney U tests with Bonferroni correction")
                
                n_comparisons = len(groups) * (len(groups) - 1) // 2
                bonferroni_alpha = alpha / n_comparisons
                
                pairwise_results = []
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        u_stat, u_p = stats.mannwhitneyu(groups[i], groups[j])
                        significant = u_p < bonferroni_alpha
                        
                        pairwise_results.append({
                            'Comparison': f"{group_names[i]} vs {group_names[j]}",
                            'U-statistic': f"{u_stat:.0f}",
                            'p-value': f"{u_p:.6f}",
                            'Significant': "✅" if significant else "❌"
                        })
                
                st.dataframe(pd.DataFrame(pairwise_results), hide_index=True)
                st.write(f"**Bonferroni-corrected α:** {bonferroni_alpha:.6f}")
    
    def _friedman_test(self, data, confidence_level):
        """Friedman test implementation"""
        st.subheader("Friedman Test")
        st.markdown("Non-parametric alternative to repeated measures ANOVA for related samples.")
        
        if data is None:
            st.warning("Please upload data to perform Friedman test")
            return
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            st.error("Need at least 3 numeric variables for Friedman test")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            selected_vars = st.multiselect(
                "Select variables (treatments/conditions):",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if len(selected_vars) < 3:
                st.warning("Select at least 3 variables")
                return
        
        with col1:
            # Prepare data for Friedman test
            clean_data = data[selected_vars].dropna()
            
            if len(clean_data) < 3:
                st.error("Need at least 3 complete observations")
                return
            
            # Perform Friedman test
            friedman_stat, p_value = stats.friedmanchisquare(*[clean_data[var] for var in selected_vars])
            
            # Calculate effect size (Kendall's W)
            n = len(clean_data)
            k = len(selected_vars)
            kendalls_w = friedman_stat / (n * (k - 1))
            
            # Display results
            st.subheader("Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Friedman χ²", f"{friedman_stat:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Kendall's W", f"{kendalls_w:.4f}")
                st.metric("Number of conditions", k)
            
            with results_col3:
                st.metric("Number of subjects", n)
                st.metric("df", k - 1)
            
            # Condition statistics
            st.subheader("Condition Statistics")
            
            # Calculate ranks for each subject
            ranks_matrix = np.zeros((n, k))
            for i in range(n):
                row_data = clean_data.iloc[i].values
                ranks_matrix[i] = stats.rankdata(row_data)
            
            condition_stats = []
            for j, var in enumerate(selected_vars):
                condition_stats.append({
                    'Condition': var,
                    'Mean': f"{clean_data[var].mean():.4f}",
                    'Median': f"{clean_data[var].median():.4f}",
                    'Mean Rank': f"{ranks_matrix[:, j].mean():.2f}",
                    'Sum of Ranks': f"{ranks_matrix[:, j].sum():.0f}",
                    'Std Dev': f"{clean_data[var].std():.4f}"
                })
            
            st.dataframe(pd.DataFrame(condition_stats), hide_index=True)
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write("There is a significant difference between conditions")
            else:
                st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write("No significant differences between conditions")
            
            # Effect size interpretation
            if kendalls_w < 0.1:
                effect_magnitude = "negligible"
            elif kendalls_w < 0.3:
                effect_magnitude = "small"
            elif kendalls_w < 0.5:
                effect_magnitude = "medium"
            else:
                effect_magnitude = "large"
            
            st.write(f"**Effect size magnitude:** {effect_magnitude}")
            st.write(f"**Kendall's W interpretation:** {kendalls_w:.3f} indicates {'strong' if kendalls_w > 0.7 else 'moderate' if kendalls_w > 0.3 else 'weak'} agreement among rankings")
            
            # Visualization
            fig = self.viz_utils.create_friedman_plot(clean_data, selected_vars)
            st.plotly_chart(fig, use_container_width=True)
    
    def _wilcoxon_rank_sum(self, data, confidence_level):
        """Wilcoxon rank-sum test (alternative name for Mann-Whitney U)"""
        st.subheader("Wilcoxon Rank-Sum Test")
        st.markdown("Alternative name for Mann-Whitney U test. Tests equality of distributions.")
        
        st.info("This is the same as the Mann-Whitney U test. Redirecting...")
        self._mann_whitney_u(data, confidence_level)
    
    def _sign_test(self, data, confidence_level):
        """Sign test implementation"""
        st.subheader("Sign Test")
        st.markdown("Tests whether the median difference equals zero using only the sign of differences.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if data is not None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    var1 = st.selectbox("Variable 1 (before):", numeric_cols)
                    var2 = st.selectbox("Variable 2 (after):", numeric_cols, index=1)
                    use_data = True
                else:
                    st.warning("Need at least 2 numeric variables")
                    use_data = False
            else:
                use_data = False
            
            if not use_data:
                st.subheader("Enter Raw Differences")
                diff_input = st.text_area(
                    "Enter differences (comma-separated):",
                    placeholder="1.2, -0.5, 2.1, -1.8, 0.3"
                )
            
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"],
                key="sign_alt"
            )
        
        with col1:
            if use_data and data is not None and var1 != var2:
                clean_data = data[[var1, var2]].dropna()
                differences = clean_data[var2] - clean_data[var1]
                non_zero_diff = differences[differences != 0]
                
            elif not use_data and diff_input:
                try:
                    differences = pd.Series([float(x.strip()) for x in diff_input.split(',')])
                    non_zero_diff = differences[differences != 0]
                    clean_data = None
                except:
                    st.error("Invalid input format")
                    return
            else:
                st.warning("Please provide data")
                return
            
            if len(non_zero_diff) == 0:
                st.error("No non-zero differences found")
                return
            
            # Perform sign test
            n_positive = sum(non_zero_diff > 0)
            n_negative = sum(non_zero_diff < 0)
            n_total = len(non_zero_diff)
            
            # Calculate p-value using binomial distribution
            if alternative == "two-sided":
                p_value = 2 * min(
                    stats.binom.cdf(n_positive, n_total, 0.5),
                    stats.binom.cdf(n_negative, n_total, 0.5)
                )
            elif alternative == "greater":
                p_value = stats.binom.cdf(n_negative, n_total, 0.5)
            else:  # less
                p_value = stats.binom.cdf(n_positive, n_total, 0.5)
            
            # Effect size (proportion of positive differences)
            effect_size = n_positive / n_total
            
            # Display results
            st.subheader("Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Positive differences", n_positive)
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Negative differences", n_negative)
                st.metric("Total differences", n_total)
            
            with results_col3:
                st.metric("Proportion positive", f"{effect_size:.4f}")
                st.metric("Median difference", f"{non_zero_diff.median():.4f}")
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write("The median difference is significantly different from zero")
            else:
                st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write("No significant difference from zero median")
            
            # Sign test is particularly useful when...
            st.subheader("Test Characteristics")
            st.write("**Sign test advantages:**")
            st.write("• Makes no assumptions about distribution shape")
            st.write("• Robust to outliers")
            st.write("• Simple to calculate and understand")
            st.write("• Only requires ordinal data")
            
            st.write("**Sign test limitations:**")
            st.write("• Less powerful than Wilcoxon signed-rank test")
            st.write("• Ignores magnitude of differences")
            st.write("• May be conservative for small samples")
            
            # Visualization
            if clean_data is not None:
                fig = self.viz_utils.create_sign_test_plot(
                    clean_data[var1], clean_data[var2], differences, var1, var2
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _runs_test(self, data, confidence_level):
        """Runs test for randomness"""
        st.subheader("Runs Test for Randomness")
        st.markdown("Tests whether a sequence of binary outcomes is random.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if data is not None:
                columns = data.columns.tolist()
                selected_col = st.selectbox("Select variable:", columns)
                
                # Option to convert to binary
                if data[selected_col].dtype in ['object', 'category']:
                    unique_vals = data[selected_col].dropna().unique()
                    if len(unique_vals) == 2:
                        binary_data = data[selected_col].dropna()
                        val1, val2 = unique_vals[0], unique_vals[1]
                        st.write(f"Binary values: {val1}, {val2}")
                    else:
                        st.error("Variable must have exactly 2 unique values")
                        return
                else:
                    # Convert numeric to binary based on median
                    numeric_data = data[selected_col].dropna()
                    median_val = numeric_data.median()
                    binary_data = numeric_data > median_val
                    val1, val2 = f"≤ {median_val:.3f}", f"> {median_val:.3f}"
                    st.write(f"Binary split at median: {val1}, {val2}")
                
                use_data = True
            else:
                use_data = False
                st.subheader("Enter Binary Sequence")
                sequence_input = st.text_area(
                    "Enter binary sequence (0s and 1s):",
                    placeholder="0,1,1,0,1,0,0,1,1,1,0"
                )
        
        with col1:
            if use_data:
                if data[selected_col].dtype in ['object', 'category']:
                    sequence = (binary_data == unique_vals[0]).astype(int)
                else:
                    sequence = binary_data.astype(int)
                    
            elif sequence_input:
                try:
                    sequence = np.array([int(x.strip()) for x in sequence_input.split(',')])
                except:
                    st.error("Invalid sequence format. Use 0s and 1s separated by commas.")
                    return
            else:
                st.warning("Please provide a binary sequence")
                return
            
            if len(sequence) < 10:
                st.warning("Runs test is more reliable with larger sequences (n ≥ 10)")
            
            # Count runs and calculate test statistic
            runs = self._count_runs(sequence)
            n1 = sum(sequence == 0)
            n2 = sum(sequence == 1)
            n = len(sequence)
            
            if n1 == 0 or n2 == 0:
                st.error("Sequence must contain both 0s and 1s")
                return
            
            # Expected runs and variance
            expected_runs = (2 * n1 * n2) / n + 1
            variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
            
            # Z-score (normal approximation)
            z_score = (runs - expected_runs) / np.sqrt(variance_runs)
            
            # P-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            # Display results
            st.subheader("Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Observed runs", runs)
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Expected runs", f"{expected_runs:.2f}")
                st.metric("Z-score", f"{z_score:.4f}")
            
            with results_col3:
                st.metric("n₁ (0s)", n1)
                st.metric("n₂ (1s)", n2)
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.error(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write("The sequence is **NOT random** - there is a significant pattern")
                if runs < expected_runs:
                    st.write("• **Too few runs**: sequence shows clustering or trend")
                else:
                    st.write("• **Too many runs**: sequence shows excessive alternation")
            else:
                st.success(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write("The sequence appears to be **random**")
            
            # Additional statistics
            st.subheader("Sequence Analysis")
            
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.write(f"**Sequence length:** {n}")
                st.write(f"**Proportion of 1s:** {n2/n:.3f}")
                st.write(f"**Longest run of 0s:** {self._longest_run(sequence, 0)}")
                st.write(f"**Longest run of 1s:** {self._longest_run(sequence, 1)}")
            
            with col_stats2:
                # Visualize sequence
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(sequence))),
                    y=sequence,
                    mode='lines+markers',
                    name='Sequence',
                    line=dict(shape='hv')
                ))
                fig.update_layout(
                    title="Binary Sequence",
                    xaxis_title="Position",
                    yaxis_title="Value",
                    yaxis=dict(tickvals=[0, 1], ticktext=['0', '1'])
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _ks_two_sample(self, data, confidence_level):
        """Kolmogorov-Smirnov two-sample test"""
        st.subheader("Kolmogorov-Smirnov Two-Sample Test")
        st.markdown("Tests whether two samples come from the same continuous distribution.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if data is not None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols and categorical_cols:
                    dependent_var = st.selectbox("Dependent variable:", numeric_cols)
                    grouping_var = st.selectbox("Grouping variable:", categorical_cols)
                    use_data = True
                else:
                    st.warning("Need both numeric and categorical variables")
                    use_data = False
            else:
                use_data = False
            
            if not use_data:
                st.subheader("Enter Raw Data")
                sample1_input = st.text_area("Sample 1 data:", placeholder="1.2, 2.3, 3.4")
                sample2_input = st.text_area("Sample 2 data:", placeholder="2.1, 3.2, 4.3")
        
        with col1:
            if use_data and data is not None:
                unique_groups = data[grouping_var].dropna().unique()
                if len(unique_groups) != 2:
                    st.error("Grouping variable must have exactly 2 categories")
                    return
                
                sample1 = data[data[grouping_var] == unique_groups[0]][dependent_var].dropna()
                sample2 = data[data[grouping_var] == unique_groups[1]][dependent_var].dropna()
                group1_name, group2_name = str(unique_groups[0]), str(unique_groups[1])
                
            elif not use_data and sample1_input and sample2_input:
                try:
                    sample1 = pd.Series([float(x.strip()) for x in sample1_input.split(',')])
                    sample2 = pd.Series([float(x.strip()) for x in sample2_input.split(',')])
                    group1_name, group2_name = "Sample 1", "Sample 2"
                except:
                    st.error("Invalid input format")
                    return
            else:
                st.warning("Please provide data for both samples")
                return
            
            # Perform KS two-sample test
            ks_statistic, p_value = stats.ks_2samp(sample1, sample2)
            
            # Effect size (difference in medians / pooled MAD)
            median_diff = sample1.median() - sample2.median()
            mad1 = stats.median_abs_deviation(sample1)
            mad2 = stats.median_abs_deviation(sample2)
            pooled_mad = np.sqrt((mad1**2 + mad2**2) / 2)
            effect_size = median_diff / pooled_mad if pooled_mad > 0 else 0
            
            # Display results
            st.subheader("Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("KS D-statistic", f"{ks_statistic:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Effect size", f"{effect_size:.4f}")
                st.metric("Median difference", f"{median_diff:.4f}")
            
            with results_col3:
                st.metric(f"{group1_name} median", f"{sample1.median():.4f}")
                st.metric(f"{group2_name} median", f"{sample2.median():.4f}")
            
            # Sample statistics
            st.subheader("Sample Statistics")
            stats_df = pd.DataFrame({
                'Sample': [group1_name, group2_name],
                'N': [len(sample1), len(sample2)],
                'Mean': [f"{sample1.mean():.4f}", f"{sample2.mean():.4f}"],
                'Median': [f"{sample1.median():.4f}", f"{sample2.median():.4f}"],
                'Std Dev': [f"{sample1.std():.4f}", f"{sample2.std():.4f}"],
                'Min': [f"{sample1.min():.4f}", f"{sample2.min():.4f}"],
                'Max': [f"{sample1.max():.4f}", f"{sample2.max():.4f}"]
            })
            st.dataframe(stats_df, hide_index=True)
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write("The two samples come from significantly different distributions")
                st.write(f"Maximum difference between CDFs: {ks_statistic:.4f}")
            else:
                st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write("No significant difference between distributions")
            
            # KS test characteristics
            st.subheader("Test Characteristics")
            st.write("**KS test advantages:**")
            st.write("• Tests entire distribution, not just location")
            st.write("• Distribution-free (non-parametric)")
            st.write("• Sensitive to differences in shape, spread, and location")
            
            st.write("**KS test considerations:**")
            st.write("• More conservative than other tests")
            st.write("• Less powerful for differences in tails only")
            st.write("• Assumes continuous distributions")
            
            # Visualization: CDFs and distributions
            fig = self.viz_utils.create_ks_plot(sample1, sample2, group1_name, group2_name)
            st.plotly_chart(fig, use_container_width=True)
    
    def _count_runs(self, sequence):
        """Count the number of runs in a binary sequence"""
        if len(sequence) == 0:
            return 0
        
        runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                runs += 1
        return runs
    
    def _longest_run(self, sequence, value):
        """Find the longest run of a specific value"""
        if len(sequence) == 0:
            return 0
        
        max_run = 0
        current_run = 0
        
        for x in sequence:
            if x == value:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run
