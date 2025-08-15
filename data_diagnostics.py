import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from .statistical_utils import StatisticalUtils
from .visualization_utils import VisualizationUtils

class DataDiagnostics:
    def __init__(self):
        self.stat_utils = StatisticalUtils()
        self.viz_utils = VisualizationUtils()
    
    def render(self, data, confidence_level):
        st.header("🔍 Comprehensive Data Diagnostics")
        
        if data is None:
            st.warning("No data available. Please upload data in the sidebar.")
            return
        
        # Data overview
        st.subheader("Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Observations", len(data))
        with col2:
            st.metric("Variables", len(data.columns))
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        with col4:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Variables", len(numeric_cols))
        
        # Display data sample
        with st.expander("Data Preview"):
            st.dataframe(data.head(10))
        
        # Variable selection
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_columns:
            st.error("No numeric variables found for analysis.")
            return
        
        selected_var = st.selectbox("Select variable for detailed diagnostics:", numeric_columns)
        
        if selected_var:
            self._detailed_variable_diagnostics(data, selected_var, confidence_level)
        
        # Multi-variable diagnostics
        if len(numeric_columns) > 1:
            st.subheader("Multi-variable Diagnostics")
            self._multivariate_diagnostics(data, numeric_columns, confidence_level)
    
    def _detailed_variable_diagnostics(self, data, variable, confidence_level):
        """Comprehensive diagnostics for a single variable"""
        values = data[variable].dropna()
        
        st.subheader(f"Detailed Diagnostics: {variable}")
        
        # Basic statistics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Distribution visualization
            fig = self.viz_utils.create_distribution_plot(values, variable)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Descriptive Statistics**")
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Skewness', 'Kurtosis'],
                'Value': [
                    len(values),
                    f"{values.mean():.4f}",
                    f"{values.std():.4f}",
                    f"{values.min():.4f}",
                    f"{values.quantile(0.25):.4f}",
                    f"{values.median():.4f}",
                    f"{values.quantile(0.75):.4f}",
                    f"{values.max():.4f}",
                    f"{stats.skew(values):.4f}",
                    f"{stats.kurtosis(values):.4f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True)
        
        # Normality tests
        st.subheader("Normality Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Shapiro-Wilk test
            if len(values) <= 5000:
                sw_stat, sw_p = stats.shapiro(values)
                st.metric("Shapiro-Wilk p-value", f"{sw_p:.6f}")
                if sw_p < (1 - confidence_level):
                    st.error("❌ Reject normality")
                else:
                    st.success("✅ Cannot reject normality")
            else:
                st.info("Sample too large for Shapiro-Wilk")
        
        with col2:
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(values, 'norm', args=(values.mean(), values.std()))
            st.metric("KS Test p-value", f"{ks_p:.6f}")
            if ks_p < (1 - confidence_level):
                st.error("❌ Reject normality")
            else:
                st.success("✅ Cannot reject normality")
        
        with col3:
            # Anderson-Darling test
            ad_result = stats.anderson(values, dist='norm')
            critical_value = ad_result.critical_values[2]  # 5% level
            st.metric("Anderson-Darling", f"{ad_result.statistic:.4f}")
            if ad_result.statistic > critical_value:
                st.error("❌ Reject normality")
            else:
                st.success("✅ Cannot reject normality")
        
        # Q-Q plot
        fig_qq = self.viz_utils.create_qq_plot(values, variable)
        st.plotly_chart(fig_qq, use_container_width=True)
        
        # Outlier detection
        st.subheader("Outlier Detection")
        
        outlier_results = self._detect_outliers(values)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("IQR Outliers", len(outlier_results['iqr_outliers']))
            if len(outlier_results['iqr_outliers']) > 0:
                st.write("Outlier values:", outlier_results['iqr_outliers'][:5])
        
        with col2:
            st.metric("Z-score Outliers (|z| > 3)", len(outlier_results['z_outliers']))
            if len(outlier_results['z_outliers']) > 0:
                st.write("Outlier values:", outlier_results['z_outliers'][:5])
        
        with col3:
            st.metric("Isolation Forest Outliers", len(outlier_results['isolation_outliers']))
            if len(outlier_results['isolation_outliers']) > 0:
                st.write("Outlier indices:", outlier_results['isolation_indices'][:5])
        
        # Outlier visualization
        fig_outliers = self.viz_utils.create_outlier_plot(values, outlier_results, variable)
        st.plotly_chart(fig_outliers, use_container_width=True)
        
        # Power analysis
        if st.checkbox("Perform Power Analysis"):
            self._power_analysis(values, confidence_level)
    
    def _detect_outliers(self, values):
        """Detect outliers using multiple methods"""
        results = {}
        
        # IQR method
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = values[(values < lower_bound) | (values > upper_bound)]
        results['iqr_outliers'] = iqr_outliers.tolist()
        
        # Z-score method
        z_scores = np.abs(stats.zscore(values))
        z_outliers = values[z_scores > 3]
        results['z_outliers'] = z_outliers.tolist()
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_pred = iso_forest.fit_predict(values.values.reshape(-1, 1))
        isolation_outliers = values[outlier_pred == -1]
        results['isolation_outliers'] = isolation_outliers.tolist()
        results['isolation_indices'] = np.where(outlier_pred == -1)[0].tolist()
        
        return results
    
    def _multivariate_diagnostics(self, data, numeric_columns, confidence_level):
        """Multi-variable diagnostic tests"""
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        corr_matrix = data[numeric_columns].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale="RdBu",
            range_color=[-1, 1]
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Homogeneity of variance tests
        if len(numeric_columns) >= 2:
            st.subheader("Homogeneity of Variance")
            
            var1 = st.selectbox("Variable 1:", numeric_columns, key="var1")
            var2 = st.selectbox("Variable 2:", numeric_columns, key="var2", index=1)
            
            if var1 != var2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Levene's test
                    levene_stat, levene_p = stats.levene(
                        data[var1].dropna(), 
                        data[var2].dropna()
                    )
                    st.metric("Levene's Test p-value", f"{levene_p:.6f}")
                    if levene_p < (1 - confidence_level):
                        st.error("❌ Unequal variances")
                    else:
                        st.success("✅ Equal variances")
                
                with col2:
                    # Bartlett's test
                    bartlett_stat, bartlett_p = stats.bartlett(
                        data[var1].dropna(), 
                        data[var2].dropna()
                    )
                    st.metric("Bartlett's Test p-value", f"{bartlett_p:.6f}")
                    if bartlett_p < (1 - confidence_level):
                        st.error("❌ Unequal variances")
                    else:
                        st.success("✅ Equal variances")
        
        # Multivariate outliers
        if len(numeric_columns) >= 2:
            st.subheader("Multivariate Outliers")
            
            # Mahalanobis distance
            clean_data = data[numeric_columns].dropna()
            if len(clean_data) > len(numeric_columns):
                try:
                    cov_matrix = np.cov(clean_data.T)
                    inv_cov = np.linalg.inv(cov_matrix)
                    mean_vec = clean_data.mean().values
                    
                    mahal_dist = []
                    for _, row in clean_data.iterrows():
                        diff = row.values - mean_vec
                        mahal_dist.append(np.sqrt(diff.T @ inv_cov @ diff))
                    
                    # Chi-square critical value
                    chi2_critical = stats.chi2.ppf(confidence_level, len(numeric_columns))
                    multivariate_outliers = np.sum(np.array(mahal_dist) > np.sqrt(chi2_critical))
                    
                    st.metric("Multivariate Outliers", multivariate_outliers)
                    
                    # Plot Mahalanobis distances
                    fig_mahal = go.Figure()
                    fig_mahal.add_trace(go.Scatter(
                        y=mahal_dist,
                        mode='markers',
                        name='Mahalanobis Distance',
                        marker=dict(color=['red' if d > np.sqrt(chi2_critical) else 'blue' for d in mahal_dist])
                    ))
                    fig_mahal.add_hline(y=np.sqrt(chi2_critical), line_dash="dash", 
                                       annotation_text="Critical Value")
                    fig_mahal.update_layout(title="Mahalanobis Distance Plot")
                    st.plotly_chart(fig_mahal, use_container_width=True)
                    
                except np.linalg.LinAlgError:
                    st.warning("Cannot compute Mahalanobis distance (singular covariance matrix)")
    
    def _power_analysis(self, values, confidence_level):
        """Perform power analysis for the variable"""
        st.subheader("Statistical Power Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input parameters for power analysis
            effect_size = st.slider("Effect Size (Cohen's d)", 0.1, 2.0, 0.5, 0.1)
            alpha = 1 - confidence_level
            
        with col2:
            # Calculate power for different sample sizes
            sample_sizes = np.arange(10, 200, 10)
            powers = []
            
            for n in sample_sizes:
                # Approximate power calculation
                se = 1 / np.sqrt(n)
                t_critical = stats.t.ppf(1 - alpha/2, n-1)
                power = 1 - stats.t.cdf(t_critical - effect_size/se, n-1)
                powers.append(power)
            
            # Plot power curve
            fig_power = go.Figure()
            fig_power.add_trace(go.Scatter(
                x=sample_sizes,
                y=powers,
                mode='lines',
                name='Power'
            ))
            fig_power.add_hline(y=0.8, line_dash="dash", annotation_text="80% Power")
            fig_power.update_layout(
                title="Power Analysis",
                xaxis_title="Sample Size",
                yaxis_title="Statistical Power"
            )
            st.plotly_chart(fig_power, use_container_width=True)
        
        # Current sample power
        current_n = len(values)
        se = 1 / np.sqrt(current_n)
        t_critical = stats.t.ppf(1 - alpha/2, current_n-1)
        current_power = 1 - stats.t.cdf(t_critical - effect_size/se, current_n-1)
        
        st.metric(f"Current Power (n={current_n})", f"{current_power:.3f}")
        
        if current_power < 0.8:
            # Calculate required sample size for 80% power
            required_n = ((stats.norm.ppf(0.8) + stats.norm.ppf(1-alpha/2)) / effect_size) ** 2
            st.warning(f"For 80% power, approximately {int(required_n)} observations needed")
