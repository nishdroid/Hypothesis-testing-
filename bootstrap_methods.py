import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .statistical_utils import StatisticalUtils
from .visualization_utils import VisualizationUtils

class BootstrapMethods:
    def __init__(self):
        self.stat_utils = StatisticalUtils()
        self.viz_utils = VisualizationUtils()
    
    def render(self, data, confidence_level, bootstrap_iterations):
        st.header("🔄 Bootstrap Methods")
        st.markdown("Robust statistical inference using resampling techniques with minimal distributional assumptions.")
        
        # Method selection
        method_type = st.selectbox(
            "Select Bootstrap Method:",
            [
                "Bootstrap Confidence Intervals",
                "Bootstrap Hypothesis Testing",
                "Bootstrap Correlation",
                "Bootstrap Regression",
                "Bootstrap ANOVA",
                "Permutation Tests",
                "Jackknife Estimation"
            ]
        )
        
        if method_type == "Bootstrap Confidence Intervals":
            self._bootstrap_confidence_intervals(data, confidence_level, bootstrap_iterations)
        elif method_type == "Bootstrap Hypothesis Testing":
            self._bootstrap_hypothesis_testing(data, confidence_level, bootstrap_iterations)
        elif method_type == "Bootstrap Correlation":
            self._bootstrap_correlation(data, confidence_level, bootstrap_iterations)
        elif method_type == "Bootstrap Regression":
            self._bootstrap_regression(data, confidence_level, bootstrap_iterations)
        elif method_type == "Bootstrap ANOVA":
            self._bootstrap_anova(data, confidence_level, bootstrap_iterations)
        elif method_type == "Permutation Tests":
            self._permutation_tests(data, confidence_level, bootstrap_iterations)
        elif method_type == "Jackknife Estimation":
            self._jackknife_estimation(data, confidence_level)
    
    def _bootstrap_confidence_intervals(self, data, confidence_level, bootstrap_iterations):
        """Bootstrap confidence intervals for various statistics"""
        st.subheader("Bootstrap Confidence Intervals")
        st.markdown("Generate confidence intervals for statistics using bootstrap resampling.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if data is not None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_var = st.selectbox("Select variable:", numeric_cols)
                    use_data = True
                else:
                    st.warning("No numeric variables found")
                    use_data = False
            else:
                use_data = False
            
            if not use_data:
                st.subheader("Enter Raw Data")
                data_input = st.text_area(
                    "Enter data (comma-separated):",
                    placeholder="1.2, 2.3, 3.4, 4.5, 5.6"
                )
            
            statistic_type = st.selectbox(
                "Statistic to bootstrap:",
                ["Mean", "Median", "Standard Deviation", "Variance", "Skewness", "Kurtosis", "IQR"]
            )
            
            method = st.selectbox(
                "Bootstrap method:",
                ["Percentile", "Bias-corrected", "BCa (Bias-corrected and accelerated)"]
            )
        
        with col1:
            if use_data and data is not None:
                values = data[selected_var].dropna()
            elif not use_data and data_input:
                try:
                    values = pd.Series([float(x.strip()) for x in data_input.split(',')])
                except:
                    st.error("Invalid data format")
                    return
            else:
                st.warning("Please provide data")
                return
            
            if len(values) < 10:
                st.warning("Bootstrap is more reliable with larger samples (n ≥ 30)")
            
            # Perform bootstrap
            with st.spinner(f"Performing {bootstrap_iterations:,} bootstrap iterations..."):
                bootstrap_stats = self._perform_bootstrap(values, statistic_type, bootstrap_iterations)
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            
            if method == "Percentile":
                ci_lower = np.percentile(bootstrap_stats, 100 * alpha/2)
                ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
            elif method == "Bias-corrected":
                ci_lower, ci_upper = self._bias_corrected_ci(
                    values, bootstrap_stats, statistic_type, confidence_level
                )
            else:  # BCa
                ci_lower, ci_upper = self._bca_ci(
                    values, bootstrap_stats, statistic_type, confidence_level
                )
            
            # Original statistic
            original_stat = self._calculate_statistic(values, statistic_type)
            
            # Display results
            st.subheader("Bootstrap Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Original Statistic", f"{original_stat:.6f}")
                st.metric("Bootstrap Mean", f"{bootstrap_stats.mean():.6f}")
            
            with results_col2:
                st.metric("Bootstrap Std", f"{bootstrap_stats.std():.6f}")
                st.metric("Bias Estimate", f"{bootstrap_stats.mean() - original_stat:.6f}")
            
            with results_col3:
                st.metric("CI Lower", f"{ci_lower:.6f}")
                st.metric("CI Upper", f"{ci_upper:.6f}")
            
            # Bootstrap distribution
            st.subheader("Bootstrap Distribution")
            
            fig = go.Figure()
            
            # Histogram of bootstrap statistics
            fig.add_trace(go.Histogram(
                x=bootstrap_stats,
                nbinsx=50,
                name="Bootstrap Distribution",
                opacity=0.7
            ))
            
            # Original statistic
            fig.add_vline(
                x=original_stat,
                line_dash="dash",
                line_color="red",
                annotation_text="Original Statistic"
            )
            
            # Confidence interval
            fig.add_vline(
                x=ci_lower,
                line_dash="dot",
                line_color="green",
                annotation_text=f"{confidence_level*100:.1f}% CI Lower"
            )
            fig.add_vline(
                x=ci_upper,
                line_dash="dot",
                line_color="green",
                annotation_text=f"{confidence_level*100:.1f}% CI Upper"
            )
            
            fig.update_layout(
                title=f"Bootstrap Distribution of {statistic_type}",
                xaxis_title=statistic_type,
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Bootstrap Summary")
            summary_stats = {
                'Statistic': ['Count', 'Mean', 'Std', 'Min', 'Q1', 'Median', 'Q3', 'Max'],
                'Value': [
                    f"{len(bootstrap_stats)}",
                    f"{bootstrap_stats.mean():.6f}",
                    f"{bootstrap_stats.std():.6f}",
                    f"{bootstrap_stats.min():.6f}",
                    f"{np.percentile(bootstrap_stats, 25):.6f}",
                    f"{np.percentile(bootstrap_stats, 50):.6f}",
                    f"{np.percentile(bootstrap_stats, 75):.6f}",
                    f"{bootstrap_stats.max():.6f}"
                ]
            }
            st.dataframe(pd.DataFrame(summary_stats), hide_index=True)
            
            # Method comparison
            if st.checkbox("Compare CI Methods"):
                self._compare_ci_methods(values, bootstrap_stats, statistic_type, confidence_level)
    
    def _bootstrap_hypothesis_testing(self, data, confidence_level, bootstrap_iterations):
        """Bootstrap hypothesis testing"""
        st.subheader("Bootstrap Hypothesis Testing")
        st.markdown("Test hypotheses using bootstrap resampling without distributional assumptions.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            test_type = st.selectbox(
                "Test type:",
                ["One-sample test", "Two-sample test", "Paired test"]
            )
            
            if data is not None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                
                if test_type == "One-sample test":
                    if numeric_cols:
                        selected_var = st.selectbox("Variable:", numeric_cols)
                        null_value = st.number_input("Null hypothesis value:", value=0.0)
                        use_data = True
                    else:
                        use_data = False
                elif test_type == "Two-sample test":
                    if numeric_cols and categorical_cols:
                        dependent_var = st.selectbox("Dependent variable:", numeric_cols)
                        grouping_var = st.selectbox("Grouping variable:", categorical_cols)
                        use_data = True
                    else:
                        use_data = False
                else:  # Paired test
                    if len(numeric_cols) >= 2:
                        var1 = st.selectbox("Variable 1:", numeric_cols)
                        var2 = st.selectbox("Variable 2:", numeric_cols, index=1)
                        use_data = True
                    else:
                        use_data = False
            else:
                use_data = False
            
            if not use_data:
                st.info("Enter summary statistics or upload data")
            
            statistic = st.selectbox(
                "Test statistic:",
                ["Mean", "Median", "Difference in means", "Difference in medians"]
            )
            
            alternative = st.selectbox(
                "Alternative hypothesis:",
                ["two-sided", "greater", "less"]
            )
        
        with col1:
            if not use_data:
                st.warning("Please upload appropriate data for the selected test")
                return
            
            # Perform bootstrap hypothesis test based on test type
            if test_type == "One-sample test":
                sample_data = data[selected_var].dropna()
                p_value, test_stat, null_dist = self._bootstrap_one_sample_test(
                    sample_data, null_value, statistic, alternative, bootstrap_iterations
                )
                
            elif test_type == "Two-sample test":
                unique_groups = data[grouping_var].dropna().unique()
                if len(unique_groups) != 2:
                    st.error("Grouping variable must have exactly 2 categories")
                    return
                
                group1 = data[data[grouping_var] == unique_groups[0]][dependent_var].dropna()
                group2 = data[data[grouping_var] == unique_groups[1]][dependent_var].dropna()
                
                p_value, test_stat, null_dist = self._bootstrap_two_sample_test(
                    group1, group2, statistic, alternative, bootstrap_iterations
                )
                
            else:  # Paired test
                clean_data = data[[var1, var2]].dropna()
                if len(clean_data) == 0:
                    st.error("No complete pairs found")
                    return
                
                p_value, test_stat, null_dist = self._bootstrap_paired_test(
                    clean_data[var1], clean_data[var2], statistic, alternative, bootstrap_iterations
                )
            
            # Display results
            st.subheader("Bootstrap Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Test Statistic", f"{test_stat:.6f}")
                st.metric("Bootstrap p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Bootstrap iterations", f"{bootstrap_iterations:,}")
                alpha = 1 - confidence_level
                st.metric("Significance level", f"{alpha:.3f}")
            
            with results_col3:
                if p_value < alpha:
                    st.success("✅ Reject H₀")
                else:
                    st.info("❌ Fail to reject H₀")
            
            # Null distribution plot
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=null_dist,
                nbinsx=50,
                name="Null Distribution",
                opacity=0.7
            ))
            
            fig.add_vline(
                x=test_stat,
                line_dash="dash",
                line_color="red",
                annotation_text="Observed Statistic"
            )
            
            # P-value regions
            if alternative == "two-sided":
                critical_lower = np.percentile(null_dist, 100 * alpha/2)
                critical_upper = np.percentile(null_dist, 100 * (1 - alpha/2))
                fig.add_vrect(
                    x0=null_dist.min(), x1=critical_lower,
                    fillcolor="red", opacity=0.2,
                    annotation_text="Rejection Region"
                )
                fig.add_vrect(
                    x0=critical_upper, x1=null_dist.max(),
                    fillcolor="red", opacity=0.2
                )
            elif alternative == "greater":
                critical_value = np.percentile(null_dist, 100 * (1 - alpha))
                fig.add_vrect(
                    x0=critical_value, x1=null_dist.max(),
                    fillcolor="red", opacity=0.2,
                    annotation_text="Rejection Region"
                )
            else:  # less
                critical_value = np.percentile(null_dist, 100 * alpha)
                fig.add_vrect(
                    x0=null_dist.min(), x1=critical_value,
                    fillcolor="red", opacity=0.2,
                    annotation_text="Rejection Region"
                )
            
            fig.update_layout(
                title=f"Bootstrap Null Distribution ({test_type})",
                xaxis_title="Test Statistic",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.subheader("Interpretation")
            if p_value < alpha:
                st.write("🔍 **Conclusion:** The bootstrap test provides evidence against the null hypothesis.")
                st.write(f"• The observed test statistic ({test_stat:.4f}) is in the extreme {p_value*100:.2f}% of the null distribution")
                st.write(f"• This is considered statistically significant at α = {alpha:.3f}")
            else:
                st.write("🔍 **Conclusion:** The bootstrap test does not provide sufficient evidence against the null hypothesis.")
                st.write(f"• The observed test statistic ({test_stat:.4f}) is not extreme enough")
                st.write(f"• The p-value ({p_value:.4f}) exceeds the significance level ({alpha:.3f})")
    
    def _bootstrap_correlation(self, data, confidence_level, bootstrap_iterations):
        """Bootstrap correlation analysis"""
        st.subheader("Bootstrap Correlation Analysis")
        st.markdown("Robust correlation analysis with bootstrap confidence intervals.")
        
        if data is None:
            st.warning("Please upload data for correlation analysis")
            return
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric variables")
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
            if var1 == var2:
                st.error("Please select different variables")
                return
            
            # Clean data
            clean_data = data[[var1, var2]].dropna()
            
            if len(clean_data) < 10:
                st.warning("Small sample size may affect bootstrap reliability")
            
            # Original correlation
            if correlation_type == "Pearson":
                original_corr, _ = stats.pearsonr(clean_data[var1], clean_data[var2])
            elif correlation_type == "Spearman":
                original_corr, _ = stats.spearmanr(clean_data[var1], clean_data[var2])
            else:  # Kendall
                original_corr, _ = stats.kendalltau(clean_data[var1], clean_data[var2])
            
            # Bootstrap correlations
            with st.spinner(f"Bootstrapping {bootstrap_iterations:,} correlations..."):
                bootstrap_corrs = []
                n = len(clean_data)
                
                for _ in range(bootstrap_iterations):
                    # Resample pairs
                    indices = np.random.choice(n, n, replace=True)
                    boot_data = clean_data.iloc[indices]
                    
                    if correlation_type == "Pearson":
                        boot_corr, _ = stats.pearsonr(boot_data[var1], boot_data[var2])
                    elif correlation_type == "Spearman":
                        boot_corr, _ = stats.spearmanr(boot_data[var1], boot_data[var2])
                    else:  # Kendall
                        boot_corr, _ = stats.kendalltau(boot_data[var1], boot_data[var2])
                    
                    bootstrap_corrs.append(boot_corr)
                
                bootstrap_corrs = np.array(bootstrap_corrs)
            
            # Confidence intervals
            alpha = 1 - confidence_level
            ci_lower = np.percentile(bootstrap_corrs, 100 * alpha/2)
            ci_upper = np.percentile(bootstrap_corrs, 100 * (1 - alpha/2))
            
            # Display results
            st.subheader("Bootstrap Correlation Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric(f"Original {correlation_type} r", f"{original_corr:.6f}")
                st.metric("Bootstrap Mean", f"{bootstrap_corrs.mean():.6f}")
            
            with results_col2:
                st.metric("Bootstrap Std", f"{bootstrap_corrs.std():.6f}")
                st.metric("Bias Estimate", f"{bootstrap_corrs.mean() - original_corr:.6f}")
            
            with results_col3:
                st.metric("CI Lower", f"{ci_lower:.6f}")
                st.metric("CI Upper", f"{ci_upper:.6f}")
            
            # Bootstrap distribution
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=bootstrap_corrs,
                nbinsx=50,
                name="Bootstrap Correlations",
                opacity=0.7
            ))
            
            fig.add_vline(
                x=original_corr,
                line_dash="dash",
                line_color="red",
                annotation_text="Original Correlation"
            )
            
            fig.add_vline(x=ci_lower, line_dash="dot", line_color="green")
            fig.add_vline(x=ci_upper, line_dash="dot", line_color="green")
            
            fig.update_layout(
                title=f"Bootstrap Distribution of {correlation_type} Correlation",
                xaxis_title="Correlation Coefficient",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Original scatter plot with bootstrap samples
            if st.checkbox("Show Bootstrap Samples"):
                fig_scatter = go.Figure()
                
                # Original data
                fig_scatter.add_trace(go.Scatter(
                    x=clean_data[var1],
                    y=clean_data[var2],
                    mode='markers',
                    name='Original Data',
                    marker=dict(size=8, opacity=0.7)
                ))
                
                # Add a few bootstrap samples
                for i in range(min(5, bootstrap_iterations)):
                    indices = np.random.choice(len(clean_data), len(clean_data), replace=True)
                    boot_sample = clean_data.iloc[indices]
                    fig_scatter.add_trace(go.Scatter(
                        x=boot_sample[var1],
                        y=boot_sample[var2],
                        mode='markers',
                        name=f'Bootstrap Sample {i+1}',
                        marker=dict(size=4, opacity=0.3)
                    ))
                
                fig_scatter.update_layout(
                    title=f"Original Data with Bootstrap Samples",
                    xaxis_title=var1,
                    yaxis_title=var2
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    def _bootstrap_regression(self, data, confidence_level, bootstrap_iterations):
        """Bootstrap regression analysis"""
        st.subheader("Bootstrap Regression Analysis")
        st.markdown("Robust regression inference with bootstrap confidence intervals for coefficients.")
        
        if data is None:
            st.warning("Please upload data for regression analysis")
            return
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric variables")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            y_var = st.selectbox("Dependent variable (Y):", numeric_cols)
            x_vars = st.multiselect(
                "Independent variables (X):",
                [col for col in numeric_cols if col != y_var],
                default=[col for col in numeric_cols if col != y_var][:1]
            )
            
            if not x_vars:
                st.warning("Select at least one independent variable")
                return
        
        with col1:
            # Clean data
            all_vars = [y_var] + x_vars
            clean_data = data[all_vars].dropna()
            
            if len(clean_data) < len(x_vars) + 5:
                st.error("Insufficient data for regression")
                return
            
            # Original regression
            from sklearn.linear_model import LinearRegression
            
            X = clean_data[x_vars].values
            y = clean_data[y_var].values
            
            original_model = LinearRegression().fit(X, y)
            original_coefs = np.concatenate([[original_model.intercept_], original_model.coef_])
            original_r2 = original_model.score(X, y)
            
            # Bootstrap regression
            with st.spinner(f"Bootstrapping {bootstrap_iterations:,} regressions..."):
                bootstrap_coefs = []
                bootstrap_r2s = []
                n = len(clean_data)
                
                for _ in range(bootstrap_iterations):
                    # Resample
                    indices = np.random.choice(n, n, replace=True)
                    boot_data = clean_data.iloc[indices]
                    
                    X_boot = boot_data[x_vars].values
                    y_boot = boot_data[y_var].values
                    
                    # Fit model
                    boot_model = LinearRegression().fit(X_boot, y_boot)
                    boot_coefs = np.concatenate([[boot_model.intercept_], boot_model.coef_])
                    boot_r2 = boot_model.score(X_boot, y_boot)
                    
                    bootstrap_coefs.append(boot_coefs)
                    bootstrap_r2s.append(boot_r2)
                
                bootstrap_coefs = np.array(bootstrap_coefs)
                bootstrap_r2s = np.array(bootstrap_r2s)
            
            # Confidence intervals
            alpha = 1 - confidence_level
            ci_lower = np.percentile(bootstrap_coefs, 100 * alpha/2, axis=0)
            ci_upper = np.percentile(bootstrap_coefs, 100 * (1 - alpha/2), axis=0)
            
            # Display results
            st.subheader("Bootstrap Regression Results")
            
            # Coefficients table
            coef_names = ['Intercept'] + x_vars
            coef_results = []
            
            for i, name in enumerate(coef_names):
                coef_results.append({
                    'Coefficient': name,
                    'Original': f"{original_coefs[i]:.6f}",
                    'Bootstrap Mean': f"{bootstrap_coefs[:, i].mean():.6f}",
                    'Bootstrap Std': f"{bootstrap_coefs[:, i].std():.6f}",
                    'CI Lower': f"{ci_lower[i]:.6f}",
                    'CI Upper': f"{ci_upper[i]:.6f}",
                    'Bias': f"{bootstrap_coefs[:, i].mean() - original_coefs[i]:.6f}"
                })
            
            st.dataframe(pd.DataFrame(coef_results), hide_index=True)
            
            # R-squared results
            r2_ci_lower = np.percentile(bootstrap_r2s, 100 * alpha/2)
            r2_ci_upper = np.percentile(bootstrap_r2s, 100 * (1 - alpha/2))
            
            st.subheader("Model Fit Statistics")
            fit_col1, fit_col2, fit_col3 = st.columns(3)
            
            with fit_col1:
                st.metric("Original R²", f"{original_r2:.6f}")
                st.metric("Bootstrap Mean R²", f"{bootstrap_r2s.mean():.6f}")
            
            with fit_col2:
                st.metric("Bootstrap Std R²", f"{bootstrap_r2s.std():.6f}")
                st.metric("R² Bias", f"{bootstrap_r2s.mean() - original_r2:.6f}")
            
            with fit_col3:
                st.metric("R² CI Lower", f"{r2_ci_lower:.6f}")
                st.metric("R² CI Upper", f"{r2_ci_upper:.6f}")
            
            # Coefficient distributions
            if len(x_vars) <= 4:  # Limit plots for readability
                fig = make_subplots(
                    rows=len(coef_names), cols=1,
                    subplot_titles=[f"Bootstrap Distribution: {name}" for name in coef_names],
                    vertical_spacing=0.08
                )
                
                for i, name in enumerate(coef_names):
                    fig.add_trace(
                        go.Histogram(
                            x=bootstrap_coefs[:, i],
                            name=name,
                            nbinsx=30,
                            showlegend=False
                        ),
                        row=i+1, col=1
                    )
                    
                    # Add original coefficient line
                    fig.add_vline(
                        x=original_coefs[i],
                        line_dash="dash",
                        line_color="red",
                        row=i+1, col=1
                    )
                
                fig.update_layout(height=200 * len(coef_names), 
                                title="Bootstrap Distributions of Regression Coefficients")
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction intervals
            if st.checkbox("Generate Prediction Intervals"):
                self._bootstrap_prediction_intervals(
                    clean_data, y_var, x_vars, bootstrap_iterations, confidence_level
                )
    
    def _bootstrap_anova(self, data, confidence_level, bootstrap_iterations):
        """Bootstrap ANOVA"""
        st.subheader("Bootstrap ANOVA")
        st.markdown("Non-parametric ANOVA using bootstrap resampling.")
        
        if data is None:
            st.warning("Please upload data for ANOVA")
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
                        groups.append(group_data.values)
                        group_names.append(str(group_name))
            
            if len(groups) < 2:
                st.error("Need at least 2 groups")
                return
            
            # Original F-statistic
            original_f, _ = stats.f_oneway(*groups)
            
            # Bootstrap ANOVA
            with st.spinner(f"Bootstrapping {bootstrap_iterations:,} F-statistics..."):
                bootstrap_f_stats = []
                
                # Pool all data for null hypothesis (no group differences)
                pooled_data = np.concatenate(groups)
                group_sizes = [len(group) for group in groups]
                
                for _ in range(bootstrap_iterations):
                    # Resample from pooled data maintaining group sizes
                    shuffled_data = np.random.choice(pooled_data, len(pooled_data), replace=True)
                    
                    # Split into groups with original sizes
                    boot_groups = []
                    start_idx = 0
                    for size in group_sizes:
                        boot_groups.append(shuffled_data[start_idx:start_idx + size])
                        start_idx += size
                    
                    # Calculate F-statistic
                    boot_f, _ = stats.f_oneway(*boot_groups)
                    bootstrap_f_stats.append(boot_f)
                
                bootstrap_f_stats = np.array(bootstrap_f_stats)
            
            # P-value (proportion of bootstrap F-stats >= original F)
            p_value = np.mean(bootstrap_f_stats >= original_f)
            
            # Display results
            st.subheader("Bootstrap ANOVA Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Original F-statistic", f"{original_f:.6f}")
                st.metric("Bootstrap p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Bootstrap Mean F", f"{bootstrap_f_stats.mean():.6f}")
                st.metric("Bootstrap Std F", f"{bootstrap_f_stats.std():.6f}")
            
            with results_col3:
                alpha = 1 - confidence_level
                if p_value < alpha:
                    st.success("✅ Reject H₀")
                else:
                    st.info("❌ Fail to reject H₀")
            
            # F-statistic distribution
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=bootstrap_f_stats,
                nbinsx=50,
                name="Bootstrap F-statistics",
                opacity=0.7
            ))
            
            fig.add_vline(
                x=original_f,
                line_dash="dash",
                line_color="red",
                annotation_text="Original F-statistic"
            )
            
            fig.update_layout(
                title="Bootstrap Distribution of F-statistics",
                xaxis_title="F-statistic",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Group statistics
            st.subheader("Group Statistics")
            group_stats = []
            for group, name in zip(groups, group_names):
                group_stats.append({
                    'Group': name,
                    'N': len(group),
                    'Mean': f"{np.mean(group):.4f}",
                    'Std': f"{np.std(group, ddof=1):.4f}",
                    'Min': f"{np.min(group):.4f}",
                    'Max': f"{np.max(group):.4f}"
                })
            
            st.dataframe(pd.DataFrame(group_stats), hide_index=True)
    
    def _permutation_tests(self, data, confidence_level, bootstrap_iterations):
        """Permutation tests"""
        st.subheader("Permutation Tests")
        st.markdown("Exact tests using permutation of group labels or paired observations.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            test_type = st.selectbox(
                "Permutation test type:",
                ["Two-sample permutation", "Paired permutation", "Independence test"]
            )
            
            if data is not None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                
                if test_type == "Two-sample permutation":
                    if numeric_cols and categorical_cols:
                        dependent_var = st.selectbox("Dependent variable:", numeric_cols)
                        grouping_var = st.selectbox("Grouping variable:", categorical_cols)
                        use_data = True
                    else:
                        use_data = False
                elif test_type == "Paired permutation":
                    if len(numeric_cols) >= 2:
                        var1 = st.selectbox("Variable 1:", numeric_cols)
                        var2 = st.selectbox("Variable 2:", numeric_cols, index=1)
                        use_data = True
                    else:
                        use_data = False
                else:  # Independence test
                    if len(categorical_cols) >= 2:
                        cat_var1 = st.selectbox("Categorical variable 1:", categorical_cols)
                        cat_var2 = st.selectbox("Categorical variable 2:", categorical_cols, index=1)
                        use_data = True
                    else:
                        use_data = False
            else:
                use_data = False
            
            statistic = st.selectbox(
                "Test statistic:",
                ["Difference in means", "Difference in medians", "t-statistic", "Chi-square"]
            )
        
        with col1:
            if not use_data:
                st.warning("Please upload appropriate data")
                return
            
            if test_type == "Two-sample permutation":
                # Two-sample permutation test
                unique_groups = data[grouping_var].dropna().unique()
                if len(unique_groups) != 2:
                    st.error("Grouping variable must have exactly 2 categories")
                    return
                
                group1 = data[data[grouping_var] == unique_groups[0]][dependent_var].dropna()
                group2 = data[data[grouping_var] == unique_groups[1]][dependent_var].dropna()
                
                # Original test statistic
                if statistic == "Difference in means":
                    original_stat = group1.mean() - group2.mean()
                elif statistic == "Difference in medians":
                    original_stat = group1.median() - group2.median()
                else:  # t-statistic
                    original_stat, _ = stats.ttest_ind(group1, group2)
                
                # Permutation test
                combined_data = np.concatenate([group1, group2])
                n1, n2 = len(group1), len(group2)
                
                with st.spinner(f"Performing {bootstrap_iterations:,} permutations..."):
                    perm_stats = []
                    
                    for _ in range(bootstrap_iterations):
                        # Permute group labels
                        shuffled_data = np.random.permutation(combined_data)
                        perm_group1 = shuffled_data[:n1]
                        perm_group2 = shuffled_data[n1:]
                        
                        if statistic == "Difference in means":
                            perm_stat = perm_group1.mean() - perm_group2.mean()
                        elif statistic == "Difference in medians":
                            perm_stat = np.median(perm_group1) - np.median(perm_group2)
                        else:  # t-statistic
                            perm_stat, _ = stats.ttest_ind(perm_group1, perm_group2)
                        
                        perm_stats.append(perm_stat)
                    
                    perm_stats = np.array(perm_stats)
                
                # P-value (two-tailed)
                p_value = np.mean(np.abs(perm_stats) >= np.abs(original_stat))
                
            elif test_type == "Paired permutation":
                # Paired permutation test
                clean_data = data[[var1, var2]].dropna()
                differences = clean_data[var2] - clean_data[var1]
                
                if statistic == "Difference in means":
                    original_stat = differences.mean()
                else:  # Difference in medians
                    original_stat = differences.median()
                
                with st.spinner(f"Performing {bootstrap_iterations:,} permutations..."):
                    perm_stats = []
                    
                    for _ in range(bootstrap_iterations):
                        # Random sign flips
                        signs = np.random.choice([-1, 1], len(differences))
                        perm_differences = differences * signs
                        
                        if statistic == "Difference in means":
                            perm_stat = perm_differences.mean()
                        else:  # Difference in medians
                            perm_stat = perm_differences.median()
                        
                        perm_stats.append(perm_stat)
                    
                    perm_stats = np.array(perm_stats)
                
                # P-value (two-tailed)
                p_value = np.mean(np.abs(perm_stats) >= np.abs(original_stat))
            
            else:
                # Independence test (Chi-square permutation)
                if cat_var1 == cat_var2:
                    st.error("Please select different categorical variables")
                    return
                
                # Create contingency table
                clean_data = data[[cat_var1, cat_var2]].dropna()
                contingency_table = pd.crosstab(clean_data[cat_var1], clean_data[cat_var2])
                
                # Original chi-square statistic
                chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                original_stat = chi2
                
                with st.spinner(f"Performing {bootstrap_iterations:,} permutations..."):
                    perm_stats = []
                    
                    for _ in range(bootstrap_iterations):
                        # Permute one variable
                        perm_data = clean_data.copy()
                        perm_data[cat_var2] = np.random.permutation(perm_data[cat_var2])
                        
                        # Calculate chi-square
                        perm_table = pd.crosstab(perm_data[cat_var1], perm_data[cat_var2])
                        perm_chi2, _, _, _ = stats.chi2_contingency(perm_table)
                        perm_stats.append(perm_chi2)
                    
                    perm_stats = np.array(perm_stats)
                
                # P-value (one-tailed for chi-square)
                p_value = np.mean(perm_stats >= original_stat)
            
            # Display results
            st.subheader("Permutation Test Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Original Statistic", f"{original_stat:.6f}")
                st.metric("Permutation p-value", f"{p_value:.6f}")
            
            with results_col2:
                st.metric("Permutations", f"{bootstrap_iterations:,}")
                alpha = 1 - confidence_level
                st.metric("Significance level", f"{alpha:.3f}")
            
            with results_col3:
                if p_value < alpha:
                    st.success("✅ Reject H₀")
                else:
                    st.info("❌ Fail to reject H₀")
            
            # Permutation distribution
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=perm_stats,
                nbinsx=50,
                name="Permutation Distribution",
                opacity=0.7
            ))
            
            fig.add_vline(
                x=original_stat,
                line_dash="dash",
                line_color="red",
                annotation_text="Observed Statistic"
            )
            
            fig.update_layout(
                title=f"Permutation Distribution ({test_type})",
                xaxis_title="Test Statistic",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _jackknife_estimation(self, data, confidence_level):
        """Jackknife estimation"""
        st.subheader("Jackknife Estimation")
        st.markdown("Leave-one-out resampling for bias estimation and confidence intervals.")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if data is not None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_var = st.selectbox("Select variable:", numeric_cols)
                    use_data = True
                else:
                    st.warning("No numeric variables found")
                    use_data = False
            else:
                use_data = False
            
            if not use_data:
                st.subheader("Enter Raw Data")
                data_input = st.text_area(
                    "Enter data (comma-separated):",
                    placeholder="1.2, 2.3, 3.4, 4.5, 5.6"
                )
            
            statistic_type = st.selectbox(
                "Statistic to jackknife:",
                ["Mean", "Median", "Standard Deviation", "Variance", "Correlation"]
            )
            
            if statistic_type == "Correlation" and use_data:
                if len(numeric_cols) >= 2:
                    corr_var = st.selectbox("Second variable for correlation:", 
                                          [col for col in numeric_cols if col != selected_var])
                else:
                    st.error("Need 2 variables for correlation")
                    return
        
        with col1:
            if use_data and data is not None:
                if statistic_type == "Correlation":
                    clean_data = data[[selected_var, corr_var]].dropna()
                    values = clean_data.values
                else:
                    values = data[selected_var].dropna().values
            elif not use_data and data_input:
                try:
                    values = np.array([float(x.strip()) for x in data_input.split(',')])
                except:
                    st.error("Invalid data format")
                    return
            else:
                st.warning("Please provide data")
                return
            
            if len(values) < 10:
                st.warning("Jackknife is more reliable with larger samples")
            
            # Original statistic
            if statistic_type == "Correlation":
                original_stat = np.corrcoef(values[:, 0], values[:, 1])[0, 1]
            else:
                original_stat = self._calculate_statistic(values, statistic_type)
            
            # Jackknife
            n = len(values)
            jackknife_stats = []
            
            for i in range(n):
                # Leave-one-out sample
                if statistic_type == "Correlation":
                    jackknife_sample = np.delete(values, i, axis=0)
                    jack_stat = np.corrcoef(jackknife_sample[:, 0], jackknife_sample[:, 1])[0, 1]
                else:
                    jackknife_sample = np.delete(values, i)
                    jack_stat = self._calculate_statistic(jackknife_sample, statistic_type)
                
                jackknife_stats.append(jack_stat)
            
            jackknife_stats = np.array(jackknife_stats)
            
            # Jackknife estimates
            jackknife_mean = jackknife_stats.mean()
            jackknife_bias = (n - 1) * (jackknife_mean - original_stat)
            bias_corrected = original_stat - jackknife_bias
            
            # Jackknife standard error
            jackknife_se = np.sqrt(((n - 1) / n) * np.sum((jackknife_stats - jackknife_mean)**2))
            
            # Confidence intervals
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, n - 1)
            ci_lower = bias_corrected - t_critical * jackknife_se
            ci_upper = bias_corrected + t_critical * jackknife_se
            
            # Display results
            st.subheader("Jackknife Results")
            
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric("Original Statistic", f"{original_stat:.6f}")
                st.metric("Jackknife Mean", f"{jackknife_mean:.6f}")
            
            with results_col2:
                st.metric("Bias Estimate", f"{jackknife_bias:.6f}")
                st.metric("Bias-corrected", f"{bias_corrected:.6f}")
            
            with results_col3:
                st.metric("Jackknife SE", f"{jackknife_se:.6f}")
                st.metric("Sample Size", f"{n}")
            
            # Confidence intervals
            st.subheader("Confidence Intervals")
            ci_col1, ci_col2 = st.columns(2)
            
            with ci_col1:
                st.metric("CI Lower", f"{ci_lower:.6f}")
            with ci_col2:
                st.metric("CI Upper", f"{ci_upper:.6f}")
            
            # Jackknife distribution
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=jackknife_stats,
                nbinsx=min(30, n//2),
                name="Jackknife Statistics",
                opacity=0.7
            ))
            
            fig.add_vline(
                x=original_stat,
                line_dash="dash",
                line_color="red",
                annotation_text="Original Statistic"
            )
            
            fig.add_vline(
                x=jackknife_mean,
                line_dash="dot",
                line_color="blue",
                annotation_text="Jackknife Mean"
            )
            
            fig.update_layout(
                title=f"Jackknife Distribution of {statistic_type}",
                xaxis_title=statistic_type,
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Jackknife vs Bootstrap comparison
            st.subheader("Method Comparison")
            st.write("**Jackknife advantages:**")
            st.write("• Deterministic (no random sampling)")
            st.write("• Good bias estimation")
            st.write("• Computationally efficient")
            
            st.write("**Jackknife limitations:**")
            st.write("• Less accurate for complex statistics")
            st.write("• Assumes smooth statistic function")
            st.write("• May underestimate variance for some statistics")
    
    # Helper methods
    def _perform_bootstrap(self, data, statistic_type, bootstrap_iterations):
        """Perform bootstrap resampling"""
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(bootstrap_iterations):
            bootstrap_sample = np.random.choice(data, n, replace=True)
            stat = self._calculate_statistic(bootstrap_sample, statistic_type)
            bootstrap_stats.append(stat)
        
        return np.array(bootstrap_stats)
    
    def _calculate_statistic(self, data, statistic_type):
        """Calculate various statistics"""
        if statistic_type == "Mean":
            return np.mean(data)
        elif statistic_type == "Median":
            return np.median(data)
        elif statistic_type == "Standard Deviation":
            return np.std(data, ddof=1)
        elif statistic_type == "Variance":
            return np.var(data, ddof=1)
        elif statistic_type == "Skewness":
            return stats.skew(data)
        elif statistic_type == "Kurtosis":
            return stats.kurtosis(data)
        elif statistic_type == "IQR":
            return np.percentile(data, 75) - np.percentile(data, 25)
        else:
            return np.mean(data)  # Default
    
    def _bias_corrected_ci(self, original_data, bootstrap_stats, statistic_type, confidence_level):
        """Bias-corrected confidence intervals"""
        original_stat = self._calculate_statistic(original_data, statistic_type)
        
        # Bias correction
        z0 = stats.norm.ppf((bootstrap_stats < original_stat).mean())
        
        alpha = 1 - confidence_level
        z_alpha_2 = stats.norm.ppf(alpha/2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha/2)
        
        # Bias-corrected percentiles
        p1 = stats.norm.cdf(2 * z0 + z_alpha_2) * 100
        p2 = stats.norm.cdf(2 * z0 + z_1_alpha_2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, p1)
        ci_upper = np.percentile(bootstrap_stats, p2)
        
        return ci_lower, ci_upper
    
    def _bca_ci(self, original_data, bootstrap_stats, statistic_type, confidence_level):
        """Bias-corrected and accelerated confidence intervals"""
        original_stat = self._calculate_statistic(original_data, statistic_type)
        
        # Bias correction
        z0 = stats.norm.ppf((bootstrap_stats < original_stat).mean())
        
        # Acceleration using jackknife
        n = len(original_data)
        jackknife_stats = []
        
        for i in range(n):
            jackknife_sample = np.delete(original_data, i)
            jack_stat = self._calculate_statistic(jackknife_sample, statistic_type)
            jackknife_stats.append(jack_stat)
        
        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = jackknife_stats.mean()
        
        # Acceleration
        acceleration = np.sum((jackknife_mean - jackknife_stats)**3) / (6 * (np.sum((jackknife_mean - jackknife_stats)**2))**1.5)
        
        alpha = 1 - confidence_level
        z_alpha_2 = stats.norm.ppf(alpha/2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha/2)
        
        # BCa percentiles
        p1 = stats.norm.cdf(z0 + (z0 + z_alpha_2)/(1 - acceleration * (z0 + z_alpha_2))) * 100
        p2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2)/(1 - acceleration * (z0 + z_1_alpha_2))) * 100
        
        ci_lower = np.percentile(bootstrap_stats, p1)
        ci_upper = np.percentile(bootstrap_stats, p2)
        
        return ci_lower, ci_upper
    
    def _compare_ci_methods(self, original_data, bootstrap_stats, statistic_type, confidence_level):
        """Compare different CI methods"""
        st.subheader("Confidence Interval Method Comparison")
        
        alpha = 1 - confidence_level
        
        # Percentile method
        perc_lower = np.percentile(bootstrap_stats, 100 * alpha/2)
        perc_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
        
        # Bias-corrected method
        bc_lower, bc_upper = self._bias_corrected_ci(original_data, bootstrap_stats, statistic_type, confidence_level)
        
        # BCa method
        bca_lower, bca_upper = self._bca_ci(original_data, bootstrap_stats, statistic_type, confidence_level)
        
        # Comparison table
        comparison_data = {
            'Method': ['Percentile', 'Bias-corrected', 'BCa'],
            'Lower CI': [f"{perc_lower:.6f}", f"{bc_lower:.6f}", f"{bca_lower:.6f}"],
            'Upper CI': [f"{perc_upper:.6f}", f"{bc_upper:.6f}", f"{bca_upper:.6f}"],
            'Interval Width': [
                f"{perc_upper - perc_lower:.6f}",
                f"{bc_upper - bc_lower:.6f}",
                f"{bca_upper - bca_lower:.6f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True)
        
        st.write("**Method recommendations:**")
        st.write("• **Percentile**: Simple, assumes unbiased bootstrap distribution")
        st.write("• **Bias-corrected**: Adjusts for bias in bootstrap distribution")
        st.write("• **BCa**: Most sophisticated, adjusts for bias and skewness")
    
    def _bootstrap_one_sample_test(self, sample, null_value, statistic, alternative, iterations):
        """Bootstrap one-sample test"""
        # Shift data to null hypothesis
        shifted_sample = sample - sample.mean() + null_value
        
        if statistic == "Mean":
            test_stat = sample.mean() - null_value
            null_dist = []
            
            for _ in range(iterations):
                boot_sample = np.random.choice(shifted_sample, len(shifted_sample), replace=True)
                boot_stat = boot_sample.mean() - null_value
                null_dist.append(boot_stat)
        else:  # Median
            test_stat = sample.median() - null_value
            null_dist = []
            
            for _ in range(iterations):
                boot_sample = np.random.choice(shifted_sample, len(shifted_sample), replace=True)
                boot_stat = np.median(boot_sample) - null_value
                null_dist.append(boot_stat)
        
        null_dist = np.array(null_dist)
        
        # Calculate p-value
        if alternative == "two-sided":
            p_value = np.mean(np.abs(null_dist) >= np.abs(test_stat))
        elif alternative == "greater":
            p_value = np.mean(null_dist >= test_stat)
        else:  # less
            p_value = np.mean(null_dist <= test_stat)
        
        return p_value, test_stat, null_dist
    
    def _bootstrap_two_sample_test(self, group1, group2, statistic, alternative, iterations):
        """Bootstrap two-sample test"""
        if statistic == "Difference in means":
            test_stat = group1.mean() - group2.mean()
        else:  # Difference in medians
            test_stat = group1.median() - group2.median()
        
        # Pool samples for null hypothesis
        pooled = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)
        
        null_dist = []
        for _ in range(iterations):
            # Resample maintaining group sizes
            boot_pooled = np.random.choice(pooled, len(pooled), replace=True)
            boot_group1 = boot_pooled[:n1]
            boot_group2 = boot_pooled[n1:n1+n2]
            
            if statistic == "Difference in means":
                boot_stat = boot_group1.mean() - boot_group2.mean()
            else:  # Difference in medians
                boot_stat = np.median(boot_group1) - np.median(boot_group2)
            
            null_dist.append(boot_stat)
        
        null_dist = np.array(null_dist)
        
        # Calculate p-value
        if alternative == "two-sided":
            p_value = np.mean(np.abs(null_dist) >= np.abs(test_stat))
        elif alternative == "greater":
            p_value = np.mean(null_dist >= test_stat)
        else:  # less
            p_value = np.mean(null_dist <= test_stat)
        
        return p_value, test_stat, null_dist
    
    def _bootstrap_paired_test(self, var1, var2, statistic, alternative, iterations):
        """Bootstrap paired test"""
        differences = var2 - var1
        
        if statistic == "Mean":
            test_stat = differences.mean()
        else:  # Median
            test_stat = differences.median()
        
        # Center differences for null hypothesis
        centered_diff = differences - differences.mean()
        
        null_dist = []
        for _ in range(iterations):
            boot_diff = np.random.choice(centered_diff, len(centered_diff), replace=True)
            
            if statistic == "Mean":
                boot_stat = boot_diff.mean()
            else:  # Median
                boot_stat = np.median(boot_diff)
            
            null_dist.append(boot_stat)
        
        null_dist = np.array(null_dist)
        
        # Calculate p-value
        if alternative == "two-sided":
            p_value = np.mean(np.abs(null_dist) >= np.abs(test_stat))
        elif alternative == "greater":
            p_value = np.mean(null_dist >= test_stat)
        else:  # less
            p_value = np.mean(null_dist <= test_stat)
        
        return p_value, test_stat, null_dist
    
    def _bootstrap_prediction_intervals(self, data, y_var, x_vars, bootstrap_iterations, confidence_level):
        """Generate bootstrap prediction intervals"""
        st.subheader("Bootstrap Prediction Intervals")
        
        from sklearn.linear_model import LinearRegression
        
        # Original model
        X = data[x_vars].values
        y = data[y_var].values
        
        # Bootstrap predictions
        n = len(data)
        bootstrap_predictions = []
        
        for _ in range(min(1000, bootstrap_iterations)):  # Limit for performance
            # Bootstrap sample
            indices = np.random.choice(n, n, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit model
            model = LinearRegression().fit(X_boot, y_boot)
            
            # Predict on original X
            predictions = model.predict(X)
            bootstrap_predictions.append(predictions)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Prediction intervals
        alpha = 1 - confidence_level
        pi_lower = np.percentile(bootstrap_predictions, 100 * alpha/2, axis=0)
        pi_upper = np.percentile(bootstrap_predictions, 100 * (1 - alpha/2), axis=0)
        
        # Plot if single predictor
        if len(x_vars) == 1:
            fig = go.Figure()
            
            # Original data
            fig.add_trace(go.Scatter(
                x=data[x_vars[0]],
                y=data[y_var],
                mode='markers',
                name='Data',
                marker=dict(color='blue')
            ))
            
            # Sort for plotting
            sort_idx = np.argsort(data[x_vars[0]])
            x_sorted = data[x_vars[0]].iloc[sort_idx]
            pi_lower_sorted = pi_lower[sort_idx]
            pi_upper_sorted = pi_upper[sort_idx]
            
            # Prediction intervals
            fig.add_trace(go.Scatter(
                x=x_sorted,
                y=pi_lower_sorted,
                line=dict(color='rgba(255,0,0,0)'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=x_sorted,
                y=pi_upper_sorted,
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                name=f'{confidence_level*100:.0f}% Prediction Interval'
            ))
            
            fig.update_layout(
                title="Bootstrap Prediction Intervals",
                xaxis_title=x_vars[0],
                yaxis_title=y_var
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.write("Prediction intervals calculated for all observations")
            st.write(f"Average interval width: {np.mean(pi_upper - pi_lower):.4f}")
