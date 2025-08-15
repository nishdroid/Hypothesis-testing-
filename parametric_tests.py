import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .statistical_utils import StatisticalUtils
from .visualization_utils import VisualizationUtils

class ParametricTests:
    def __init__(self):
        self.stat_utils = StatisticalUtils()
        self.viz_utils = VisualizationUtils()
    
    def render(self, data, confidence_level, bootstrap_iterations):
        st.header("📈 Parametric Statistical Tests")
        st.markdown("Comprehensive parametric testing with assumption checking and effect size calculations.")
        
        # Test selection
        test_type = st.selectbox(
            "Select Test Type:",
            [
                "One-sample t-test",
                "Independent samples t-test", 
                "Paired samples t-test",
                "One-way ANOVA",
                "Two-way ANOVA",
                "Correlation analysis",
                "Linear regression"
            ]
        )
        
        if test_type == "One-sample t-test":
            self._one_sample_ttest(data, confidence_level)
        elif test_type == "Independent samples t-test":
            self._independent_ttest(data, confidence_level)
        elif test_type == "Paired samples t-test":
            self._paired_ttest(data, confidence_level)
        elif test_type == "One-way ANOVA":
            self._one_way_anova(data, confidence_level)
        elif test_type == "Two-way ANOVA":
            self._two_way_anova(data, confidence_level)
        elif test_type == "Correlation analysis":
            self._correlation_analysis(data, confidence_level)
        elif test_type == "Linear regression":
            self._linear_regression(data, confidence_level)
    
    def _one_sample_ttest(self, data, confidence_level):
        """One-sample t-test implementation"""
        st.subheader("One-Sample t-test")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["Raw Data", "Summary Statistics"],
                key="one_sample_input_method"
            )
            
            # Initialize variables
            selected_var = None
            sample_mean = sample_std = sample_size = None
            use_data = False
            
            if input_method == "Raw Data":
                if data is not None:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        selected_var = st.selectbox("Select variable:", numeric_cols, key="one_sample_var")
                        use_data = True
                    else:
                        st.warning("No numeric variables found. Switch to Summary Statistics.")
                else:
                    st.warning("No data uploaded. Switch to Summary Statistics.")
            else:  # Summary Statistics
                st.subheader("📊 Summary Statistics Input")
                sample_mean = st.number_input("Sample mean:", value=0.0, key="one_sample_mean")
                sample_std = st.number_input("Sample standard deviation:", value=1.0, min_value=0.01, key="one_sample_std")
                sample_size = st.number_input("Sample size:", value=30, min_value=1, key="one_sample_n")
            
            population_mean = st.number_input("Population mean (H₀):", value=0.0)
            
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"]
            )
        
        with col1:
            if use_data and data is not None and selected_var:
                values = data[selected_var].dropna()
                
                # Check assumptions
                st.subheader("Assumption Checking")
                self._check_normality(values, confidence_level)
                
                # Perform test
                t_stat, p_value = stats.ttest_1samp(values, population_mean, alternative=alternative)
                
                # Calculate effect size (Cohen's d)
                cohens_d = (values.mean() - population_mean) / values.std(ddof=1)
                
                # Confidence interval
                alpha = 1 - confidence_level
                se = values.std(ddof=1) / np.sqrt(len(values))
                t_critical = stats.t.ppf(1 - alpha/2, len(values) - 1)
                ci_lower = values.mean() - t_critical * se
                ci_upper = values.mean() + t_critical * se
                
                sample_mean = values.mean()
                sample_std = values.std(ddof=1)
                sample_size = len(values)
                
            elif input_method == "Summary Statistics":
                # Calculate from summary statistics
                se = sample_std / np.sqrt(sample_size)
                t_stat = (sample_mean - population_mean) / se
                
                df = sample_size - 1
                if alternative == "two-sided":
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                elif alternative == "greater":
                    p_value = 1 - stats.t.cdf(t_stat, df)
                else:  # less
                    p_value = stats.t.cdf(t_stat, df)
                
                cohens_d = (sample_mean - population_mean) / sample_std
                
                # Confidence interval
                alpha = 1 - confidence_level
                t_critical = stats.t.ppf(1 - alpha/2, df)
                ci_lower = sample_mean - t_critical * se
                ci_upper = sample_mean + t_critical * se
                
                values = None  # No raw data for visualization
            else:
                st.warning("Please select a variable or enter summary statistics")
                return
            
            # Display results
            st.subheader("Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("t-statistic", f"{t_stat:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Cohen's d", f"{cohens_d:.4f}")
                st.metric("Sample mean", f"{sample_mean:.4f}")
            
            with results_col3:
                st.metric("CI Lower", f"{ci_lower:.4f}")
                st.metric("CI Upper", f"{ci_upper:.4f}")
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write(f"The sample mean ({sample_mean:.4f}) is significantly different from {population_mean}")
            else:
                st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write(f"No significant difference between sample mean and {population_mean}")
            
            # Effect size interpretation
            self._interpret_effect_size(cohens_d, "Cohen's d")
            
            # Visualizations
            if values is not None:
                fig = self.viz_utils.create_one_sample_plot(values, population_mean, selected_var)
                st.plotly_chart(fig, use_container_width=True)
    
    def _independent_ttest(self, data, confidence_level):
        """Independent samples t-test implementation"""
        st.subheader("Independent Samples t-test")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["Raw Data", "Summary Statistics"],
                key="indep_input_method"
            )
            
            # Initialize variables
            dependent_var = grouping_var = None
            mean1 = std1 = n1 = mean2 = std2 = n2 = None
            use_data = False
            
            if input_method == "Raw Data":
                if data is not None:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                    
                    if numeric_cols and categorical_cols:
                        dependent_var = st.selectbox("Dependent variable:", numeric_cols, key="indep_dep_var")
                        grouping_var = st.selectbox("Grouping variable:", categorical_cols, key="indep_group_var")
                        use_data = True
                    else:
                        st.warning("Need both numeric and categorical variables. Switch to Summary Statistics.")
                else:
                    st.warning("No data uploaded. Switch to Summary Statistics.")
            else:  # Summary Statistics
                st.subheader("📊 Summary Statistics Input")
                st.write("**Group 1:**")
                mean1 = st.number_input("Group 1 mean:", value=0.0, key="indep_mean1")
                std1 = st.number_input("Group 1 std:", value=1.0, min_value=0.01, key="indep_std1")
                n1 = st.number_input("Group 1 size:", value=30, min_value=1, key="indep_n1")
                
                st.write("**Group 2:**")
                mean2 = st.number_input("Group 2 mean:", value=0.0, key="indep_mean2")
                std2 = st.number_input("Group 2 std:", value=1.0, min_value=0.01, key="indep_std2")
                n2 = st.number_input("Group 2 size:", value=30, min_value=1, key="indep_n2")
            
            equal_var = st.checkbox("Assume equal variances", value=True)
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"],
                key="indep_alt"
            )
        
        with col1:
            if use_data and data is not None:
                # Get unique groups
                unique_groups = data[grouping_var].unique()
                if len(unique_groups) != 2:
                    st.error("Grouping variable must have exactly 2 categories")
                    return
                
                group1_data = data[data[grouping_var] == unique_groups[0]][dependent_var].dropna()
                group2_data = data[data[grouping_var] == unique_groups[1]][dependent_var].dropna()
                
                # Check assumptions
                st.subheader("Assumption Checking")
                
                # Normality for both groups
                st.write("**Normality Tests:**")
                col_norm1, col_norm2 = st.columns(2)
                with col_norm1:
                    st.write(f"Group 1 ({unique_groups[0]}):")
                    self._check_normality(group1_data, confidence_level, show_title=False)
                with col_norm2:
                    st.write(f"Group 2 ({unique_groups[1]}):")
                    self._check_normality(group2_data, confidence_level, show_title=False)
                
                # Equal variances test
                st.write("**Equal Variances Test:**")
                levene_stat, levene_p = stats.levene(group1_data, group2_data)
                st.write(f"Levene's test p-value: {levene_p:.6f}")
                if levene_p < (1 - confidence_level):
                    st.warning("❌ Unequal variances detected - consider using Welch's t-test")
                    if equal_var:
                        st.info("Switching to unequal variances assumption")
                        equal_var = False
                else:
                    st.success("✅ Equal variances assumption satisfied")
                
                # Perform test
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, 
                                                equal_var=equal_var, alternative=alternative)
                
                # Calculate effect size
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var(ddof=1) + 
                                    (len(group2_data) - 1) * group2_data.var(ddof=1)) / 
                                   (len(group1_data) + len(group2_data) - 2))
                cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
                
                mean1, std1, n1 = group1_data.mean(), group1_data.std(ddof=1), len(group1_data)
                mean2, std2, n2 = group2_data.mean(), group2_data.std(ddof=1), len(group2_data)
                
            else:
                # Calculate from summary statistics
                if equal_var:
                    # Pooled variance
                    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
                    se = np.sqrt(pooled_var * (1/n1 + 1/n2))
                    df = n1 + n2 - 2
                else:
                    # Welch's t-test
                    se = np.sqrt(std1**2/n1 + std2**2/n2)
                    df = (std1**2/n1 + std2**2/n2)**2 / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
                
                t_stat = (mean1 - mean2) / se
                
                if alternative == "two-sided":
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                elif alternative == "greater":
                    p_value = 1 - stats.t.cdf(t_stat, df)
                else:  # less
                    p_value = stats.t.cdf(t_stat, df)
                
                # Effect size
                pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                cohens_d = (mean1 - mean2) / pooled_std
                
                group1_data = group2_data = None
            
            # Display results
            st.subheader("Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("t-statistic", f"{t_stat:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Cohen's d", f"{cohens_d:.4f}")
                st.metric("Mean difference", f"{mean1 - mean2:.4f}")
            
            with results_col3:
                st.metric(f"Group 1 mean", f"{mean1:.4f}")
                st.metric(f"Group 2 mean", f"{mean2:.4f}")
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write("There is a significant difference between the groups")
            else:
                st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write("No significant difference between the groups")
            
            self._interpret_effect_size(cohens_d, "Cohen's d")
            
            # Visualizations
            if group1_data is not None and group2_data is not None:
                fig = self.viz_utils.create_two_sample_plot(
                    group1_data, group2_data, 
                    unique_groups[0], unique_groups[1], 
                    dependent_var
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _paired_ttest(self, data, confidence_level):
        """Paired samples t-test implementation"""
        st.subheader("Paired Samples t-test")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["Raw Data", "Summary Statistics"],
                key="paired_input_method"
            )
            
            # Initialize variables
            var1 = var2 = None
            mean_diff = std_diff = n_pairs = None
            use_data = False
            
            if input_method == "Raw Data":
                if data is not None:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) >= 2:
                        var1 = st.selectbox("Variable 1 (before):", numeric_cols, key="paired_var1")
                        var2 = st.selectbox("Variable 2 (after):", numeric_cols, index=1, key="paired_var2")
                        use_data = True
                    else:
                        st.warning("Need at least 2 numeric variables. Switch to Summary Statistics.")
                else:
                    st.warning("No data uploaded. Switch to Summary Statistics.")
            else:  # Summary Statistics
                st.subheader("📊 Summary Statistics Input")
                mean_diff = st.number_input("Mean difference (after - before):", value=0.0, key="paired_mean_diff")
                std_diff = st.number_input("Std of differences:", value=1.0, min_value=0.01, key="paired_std_diff")
                n_pairs = st.number_input("Number of pairs:", value=30, min_value=1, key="paired_n")
            
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"],
                key="paired_alt"
            )
        
        with col1:
            if use_data and data is not None and var1 != var2:
                # Remove rows with missing values in either variable
                clean_data = data[[var1, var2]].dropna()
                
                if len(clean_data) == 0:
                    st.error("No complete pairs found")
                    return
                
                differences = clean_data[var2] - clean_data[var1]
                
                # Check normality of differences
                st.subheader("Assumption Checking")
                st.write("**Normality of Differences:**")
                self._check_normality(differences, confidence_level, show_title=False)
                
                # Perform test
                t_stat, p_value = stats.ttest_rel(clean_data[var2], clean_data[var1], 
                                                alternative=alternative)
                
                # Effect size
                cohens_d = differences.mean() / differences.std(ddof=1)
                
                mean_diff = differences.mean()
                std_diff = differences.std(ddof=1)
                n_pairs = len(differences)
                
            else:
                # Calculate from summary statistics
                se = std_diff / np.sqrt(n_pairs)
                t_stat = mean_diff / se
                df = n_pairs - 1
                
                if alternative == "two-sided":
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                elif alternative == "greater":
                    p_value = 1 - stats.t.cdf(t_stat, df)
                else:  # less
                    p_value = stats.t.cdf(t_stat, df)
                
                cohens_d = mean_diff / std_diff
                differences = None
            
            # Display results
            st.subheader("Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("t-statistic", f"{t_stat:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Cohen's d", f"{cohens_d:.4f}")
                st.metric("Mean difference", f"{mean_diff:.4f}")
            
            with results_col3:
                st.metric("Std of differences", f"{std_diff:.4f}")
                st.metric("Number of pairs", f"{n_pairs}")
            
            # Confidence interval for mean difference
            alpha = 1 - confidence_level
            se = std_diff / np.sqrt(n_pairs)
            t_critical = stats.t.ppf(1 - alpha/2, n_pairs - 1)
            ci_lower = mean_diff - t_critical * se
            ci_upper = mean_diff + t_critical * se
            
            st.write(f"**{confidence_level*100:.1f}% Confidence Interval for Mean Difference:**")
            st.write(f"[{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Interpretation
            if p_value < alpha:
                st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write("There is a significant difference between the paired measurements")
            else:
                st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write("No significant difference between the paired measurements")
            
            self._interpret_effect_size(cohens_d, "Cohen's d")
            
            # Visualizations
            if differences is not None:
                fig = self.viz_utils.create_paired_plot(
                    clean_data[var1], clean_data[var2], differences, var1, var2
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _one_way_anova(self, data, confidence_level):
        """One-way ANOVA implementation"""
        st.subheader("One-way ANOVA")
        
        if data is None:
            st.warning("Please upload data to perform ANOVA")
            return
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_cols or not categorical_cols:
            st.error("Need both numeric and categorical variables for ANOVA")
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
            
            if len(groups) < 2:
                st.error("Need at least 2 groups for ANOVA")
                return
            
            # Check assumptions
            st.subheader("Assumption Checking")
            
            # Normality for each group
            st.write("**Normality Tests by Group:**")
            for i, (group, name) in enumerate(zip(groups, group_names)):
                with st.expander(f"Group: {name}"):
                    self._check_normality(group, confidence_level, show_title=False)
            
            # Homogeneity of variance
            st.write("**Homogeneity of Variance:**")
            levene_stat, levene_p = stats.levene(*groups)
            st.write(f"Levene's test p-value: {levene_p:.6f}")
            if levene_p < (1 - confidence_level):
                st.warning("❌ Unequal variances detected - consider Welch's ANOVA")
            else:
                st.success("✅ Equal variances assumption satisfied")
            
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Calculate effect size (eta-squared)
            grand_mean = np.concatenate(groups).mean()
            ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in groups)
            ss_total = sum((np.concatenate(groups) - grand_mean)**2)
            eta_squared = ss_between / ss_total
            
            # Display results
            st.subheader("ANOVA Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("F-statistic", f"{f_stat:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("η² (eta-squared)", f"{eta_squared:.4f}")
                st.metric("Number of groups", len(groups))
            
            with results_col3:
                total_n = sum(len(group) for group in groups)
                st.metric("Total observations", total_n)
                st.metric("df between", len(groups) - 1)
            
            # Group descriptives
            st.subheader("Group Descriptives")
            descriptives = []
            for group, name in zip(groups, group_names):
                descriptives.append({
                    'Group': name,
                    'n': len(group),
                    'Mean': f"{group.mean():.4f}",
                    'Std': f"{group.std(ddof=1):.4f}",
                    'Min': f"{group.min():.4f}",
                    'Max': f"{group.max():.4f}"
                })
            
            st.dataframe(pd.DataFrame(descriptives), hide_index=True)
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success(f"✅ Reject H₀ at α = {alpha:.3f}")
                st.write("There is a significant difference between at least two groups")
                
                # Post-hoc tests
                st.subheader("Post-hoc Analysis")
                st.info("Significant ANOVA detected. Consider running post-hoc tests to identify which groups differ.")
            else:
                st.info(f"❌ Fail to reject H₀ at α = {alpha:.3f}")
                st.write("No significant differences between groups")
            
            self._interpret_effect_size(eta_squared, "η² (eta-squared)")
            
            # Visualization
            fig = self.viz_utils.create_anova_plot(groups, group_names, dependent_var)
            st.plotly_chart(fig, use_container_width=True)
    
    def _two_way_anova(self, data, confidence_level):
        """Two-way ANOVA implementation"""
        st.subheader("Two-way ANOVA")
        st.info("Two-way ANOVA requires advanced statistical computation. This is a simplified implementation.")
        
        if data is None:
            st.warning("Please upload data to perform two-way ANOVA")
            return
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_cols or len(categorical_cols) < 2:
            st.error("Need numeric variable and at least 2 categorical variables")
            return
        
        dependent_var = st.selectbox("Dependent variable:", numeric_cols)
        factor1 = st.selectbox("Factor 1:", categorical_cols)
        factor2 = st.selectbox("Factor 2:", categorical_cols, index=1)
        
        if factor1 == factor2:
            st.error("Please select different factors")
            return
        
        # Create interaction term and perform analysis
        clean_data = data[[dependent_var, factor1, factor2]].dropna()
        
        # Group means for visualization
        group_means = clean_data.groupby([factor1, factor2])[dependent_var].agg(['mean', 'count', 'std']).reset_index()
        
        st.subheader("Group Means")
        st.dataframe(group_means)
        
        # Interaction plot
        fig = px.line(
            clean_data.groupby([factor1, factor2])[dependent_var].mean().reset_index(),
            x=factor1, y=dependent_var, color=factor2,
            title=f"Interaction Plot: {dependent_var} by {factor1} and {factor2}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("For complete two-way ANOVA with interaction effects, consider using specialized statistical software.")
    
    def _correlation_analysis(self, data, confidence_level):
        """Correlation analysis implementation"""
        st.subheader("Correlation Analysis")
        
        if data is None:
            st.warning("Please upload data for correlation analysis")
            return
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric variables for correlation")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            var1 = st.selectbox("Variable 1:", numeric_cols)
            var2 = st.selectbox("Variable 2:", numeric_cols, index=1)
            
            correlation_type = st.selectbox(
                "Correlation type:",
                ["Pearson", "Spearman", "Kendall"]
            )
        
        with col1:
            if var1 != var2:
                # Get clean data
                clean_data = data[[var1, var2]].dropna()
                
                if len(clean_data) < 3:
                    st.error("Need at least 3 observations for correlation")
                    return
                
                # Calculate correlation
                if correlation_type == "Pearson":
                    corr_coef, p_value = stats.pearsonr(clean_data[var1], clean_data[var2])
                elif correlation_type == "Spearman":
                    corr_coef, p_value = stats.spearmanr(clean_data[var1], clean_data[var2])
                else:  # Kendall
                    corr_coef, p_value = stats.kendalltau(clean_data[var1], clean_data[var2])
                
                # Display results
                st.subheader("Correlation Results")
                
                results_col1, results_col2, results_col3 = st.columns(3)
                
                with results_col1:
                    st.metric(f"{correlation_type} r", f"{corr_coef:.4f}")
                    st.metric("p-value", f"{p_value:.6f}")
                
                with results_col2:
                    st.metric("r²", f"{corr_coef**2:.4f}")
                    st.metric("Sample size", len(clean_data))
                
                with results_col3:
                    # Confidence interval for correlation
                    alpha = 1 - confidence_level
                    z_critical = stats.norm.ppf(1 - alpha/2)
                    
                    # Fisher's z-transformation
                    z_r = 0.5 * np.log((1 + corr_coef) / (1 - corr_coef))
                    se_z = 1 / np.sqrt(len(clean_data) - 3)
                    
                    z_lower = z_r - z_critical * se_z
                    z_upper = z_r + z_critical * se_z
                    
                    # Transform back
                    ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                    ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                    
                    st.metric("CI Lower", f"{ci_lower:.4f}")
                    st.metric("CI Upper", f"{ci_upper:.4f}")
                
                # Interpretation
                alpha = 1 - confidence_level
                if p_value < alpha:
                    st.success(f"✅ Significant correlation at α = {alpha:.3f}")
                else:
                    st.info(f"❌ No significant correlation at α = {alpha:.3f}")
                
                # Correlation strength interpretation
                abs_corr = abs(corr_coef)
                if abs_corr < 0.1:
                    strength = "negligible"
                elif abs_corr < 0.3:
                    strength = "weak"
                elif abs_corr < 0.5:
                    strength = "moderate"
                elif abs_corr < 0.7:
                    strength = "strong"
                else:
                    strength = "very strong"
                
                direction = "positive" if corr_coef > 0 else "negative"
                st.write(f"**Correlation strength:** {strength} {direction} correlation")
                
                # Scatter plot
                fig = px.scatter(
                    clean_data, x=var1, y=var2,
                    title=f"{correlation_type} Correlation: r = {corr_coef:.3f}",
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _linear_regression(self, data, confidence_level):
        """Linear regression analysis"""
        st.subheader("Linear Regression")
        
        if data is None:
            st.warning("Please upload data for regression analysis")
            return
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric variables for regression")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            y_var = st.selectbox("Dependent variable (Y):", numeric_cols)
            x_var = st.selectbox("Independent variable (X):", 
                                [col for col in numeric_cols if col != y_var])
        
        with col1:
            # Get clean data
            clean_data = data[[y_var, x_var]].dropna()
            
            if len(clean_data) < 3:
                st.error("Need at least 3 observations for regression")
                return
            
            # Perform regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                clean_data[x_var], clean_data[y_var]
            )
            
            # Calculate additional statistics
            y_pred = slope * clean_data[x_var] + intercept
            residuals = clean_data[y_var] - y_pred
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            
            # Display results
            st.subheader("Regression Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Slope (β₁)", f"{slope:.4f}")
                st.metric("Intercept (β₀)", f"{intercept:.4f}")
            
            with results_col2:
                st.metric("R²", f"{r_value**2:.4f}")
                st.metric("p-value", f"{p_value:.6f}")
            
            with results_col3:
                st.metric("RMSE", f"{rmse:.4f}")
                st.metric("Std Error", f"{std_err:.4f}")
            
            # Regression equation
            st.write(f"**Regression Equation:** {y_var} = {intercept:.4f} + {slope:.4f} × {x_var}")
            
            # Interpretation
            alpha = 1 - confidence_level
            if p_value < alpha:
                st.success(f"✅ Significant regression at α = {alpha:.3f}")
                st.write(f"For each unit increase in {x_var}, {y_var} changes by {slope:.4f} units")
            else:
                st.info(f"❌ No significant regression at α = {alpha:.3f}")
            
            # Regression plot with confidence intervals
            fig = px.scatter(clean_data, x=x_var, y=y_var, 
                           title=f"Linear Regression: R² = {r_value**2:.3f}")
            
            # Add regression line
            x_range = np.linspace(clean_data[x_var].min(), clean_data[x_var].max(), 100)
            y_line = slope * x_range + intercept
            
            fig.add_trace(go.Scatter(x=x_range, y=y_line, mode='lines', 
                                   name='Regression Line', line=dict(color='red')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual analysis
            st.subheader("Residual Analysis")
            
            fig_residuals = make_subplots(rows=1, cols=2, 
                                        subplot_titles=['Residuals vs Fitted', 'Q-Q Plot of Residuals'])
            
            # Residuals vs fitted
            fig_residuals.add_trace(
                go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'),
                row=1, col=1
            )
            fig_residuals.add_hline(y=0, line_dash="dash", row=1, col=1)
            
            # Q-Q plot of residuals
            (osm, osr), (slope_qq, intercept_qq, r_qq) = stats.probplot(residuals, dist="norm")
            fig_residuals.add_trace(
                go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q Plot'),
                row=1, col=2
            )
            fig_residuals.add_trace(
                go.Scatter(x=osm, y=slope_qq * osm + intercept_qq, mode='lines', name='Q-Q Line'),
                row=1, col=2
            )
            
            fig_residuals.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_residuals, use_container_width=True)
    
    def _check_normality(self, data, confidence_level, show_title=True):
        """Check normality assumption"""
        if show_title:
            st.write("**Normality Tests:**")
        
        alpha = 1 - confidence_level
        
        # Shapiro-Wilk test (for smaller samples)
        if len(data) <= 5000:
            sw_stat, sw_p = stats.shapiro(data)
            st.write(f"Shapiro-Wilk p-value: {sw_p:.6f}")
            if sw_p < alpha:
                st.write("❌ Reject normality (Shapiro-Wilk)")
            else:
                st.write("✅ Cannot reject normality (Shapiro-Wilk)")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        st.write(f"KS test p-value: {ks_p:.6f}")
        if ks_p < alpha:
            st.write("❌ Reject normality (KS)")
        else:
            st.write("✅ Cannot reject normality (KS)")
    
    def _interpret_effect_size(self, effect_size, effect_type):
        """Interpret effect size magnitude"""
        st.write(f"**{effect_type} Interpretation:**")
        
        if effect_type == "Cohen's d":
            abs_effect = abs(effect_size)
            if abs_effect < 0.2:
                magnitude = "negligible"
            elif abs_effect < 0.5:
                magnitude = "small"
            elif abs_effect < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
        elif effect_type == "η² (eta-squared)":
            if effect_size < 0.01:
                magnitude = "negligible"
            elif effect_size < 0.06:
                magnitude = "small"
            elif effect_size < 0.14:
                magnitude = "medium"
            else:
                magnitude = "large"
        else:
            magnitude = "unknown"
        
        st.write(f"Effect size magnitude: **{magnitude}**")
