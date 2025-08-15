import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class StatisticalUtils:
    """Utility class for common statistical calculations and operations"""
    
    def __init__(self):
        pass
    
    # Descriptive Statistics
    def calculate_basic_stats(self, data):
        """Calculate basic descriptive statistics"""
        if len(data) == 0:
            return {}
        
        return {
            'count': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'mode': stats.mode(data)[0] if len(stats.mode(data)[0]) > 0 else np.nan,
            'std': np.std(data, ddof=1),
            'var': np.var(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
    
    def calculate_confidence_interval(self, data, confidence_level=0.95):
        """Calculate confidence interval for the mean"""
        n = len(data)
        if n < 2:
            return None, None
        
        mean = np.mean(data)
        se = stats.sem(data)
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        
        margin = t_critical * se
        return mean - margin, mean + margin
    
    def calculate_prediction_interval(self, data, confidence_level=0.95):
        """Calculate prediction interval for future observations"""
        n = len(data)
        if n < 2:
            return None, None
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        
        margin = t_critical * std * np.sqrt(1 + 1/n)
        return mean - margin, mean + margin
    
    # Normality Tests
    def test_normality(self, data, alpha=0.05):
        """Comprehensive normality testing"""
        results = {}
        
        if len(data) < 3:
            return {"error": "Insufficient data for normality tests"}
        
        # Shapiro-Wilk test (best for small to medium samples)
        if len(data) <= 5000:
            try:
                stat, p_value = stats.shapiro(data)
                results['shapiro_wilk'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > alpha
                }
            except:
                results['shapiro_wilk'] = {"error": "Could not compute Shapiro-Wilk test"}
        
        # Kolmogorov-Smirnov test
        try:
            stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            results['kolmogorov_smirnov'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > alpha
            }
        except:
            results['kolmogorov_smirnov'] = {"error": "Could not compute KS test"}
        
        # Anderson-Darling test
        try:
            result = stats.anderson(data, dist='norm')
            # Use 5% significance level (index 2)
            critical_value = result.critical_values[2]
            is_normal = result.statistic <= critical_value
            results['anderson_darling'] = {
                'statistic': result.statistic,
                'critical_value': critical_value,
                'is_normal': is_normal
            }
        except:
            results['anderson_darling'] = {"error": "Could not compute Anderson-Darling test"}
        
        # D'Agostino-Pearson test
        if len(data) >= 8:
            try:
                stat, p_value = stats.normaltest(data)
                results['dagostino_pearson'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > alpha
                }
            except:
                results['dagostino_pearson'] = {"error": "Could not compute D'Agostino-Pearson test"}
        
        return results
    
    # Outlier Detection
    def detect_outliers_iqr(self, data, factor=1.5):
        """Detect outliers using IQR method"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
        
        return {
            'outliers': outliers,
            'indices': outlier_indices,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'method': 'IQR'
        }
    
    def detect_outliers_zscore(self, data, threshold=3):
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(data))
        outlier_indices = np.where(z_scores > threshold)[0]
        outliers = data[outlier_indices]
        
        return {
            'outliers': outliers,
            'indices': outlier_indices,
            'z_scores': z_scores,
            'threshold': threshold,
            'method': 'Z-score'
        }
    
    def detect_outliers_modified_zscore(self, data, threshold=3.5):
        """Detect outliers using modified Z-score (median-based)"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return {
                'outliers': np.array([]),
                'indices': np.array([]),
                'modified_z_scores': np.zeros_like(data),
                'method': 'Modified Z-score'
            }
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
        outliers = data[outlier_indices]
        
        return {
            'outliers': outliers,
            'indices': outlier_indices,
            'modified_z_scores': modified_z_scores,
            'threshold': threshold,
            'method': 'Modified Z-score'
        }
    
    def detect_outliers_isolation_forest(self, data, contamination=0.1):
        """Detect outliers using Isolation Forest"""
        if len(data) < 10:
            return {
                'outliers': np.array([]),
                'indices': np.array([]),
                'scores': np.array([]),
                'method': 'Isolation Forest',
                'warning': 'Insufficient data for Isolation Forest'
            }
        
        # Reshape for sklearn
        data_reshaped = data.reshape(-1, 1)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data_reshaped)
        scores = iso_forest.score_samples(data_reshaped)
        
        outlier_indices = np.where(outlier_labels == -1)[0]
        outliers = data[outlier_indices]
        
        return {
            'outliers': outliers,
            'indices': outlier_indices,
            'scores': scores,
            'labels': outlier_labels,
            'contamination': contamination,
            'method': 'Isolation Forest'
        }
    
    # Variance Tests
    def test_equal_variances(self, *groups, alpha=0.05):
        """Test for equal variances across groups"""
        results = {}
        
        if len(groups) < 2:
            return {"error": "Need at least 2 groups"}
        
        # Levene's test (robust to non-normality)
        try:
            stat, p_value = stats.levene(*groups)
            results['levene'] = {
                'statistic': stat,
                'p_value': p_value,
                'equal_variances': p_value > alpha
            }
        except:
            results['levene'] = {"error": "Could not compute Levene's test"}
        
        # Bartlett's test (assumes normality)
        try:
            stat, p_value = stats.bartlett(*groups)
            results['bartlett'] = {
                'statistic': stat,
                'p_value': p_value,
                'equal_variances': p_value > alpha
            }
        except:
            results['bartlett'] = {"error": "Could not compute Bartlett's test"}
        
        # Fligner-Killeen test (non-parametric)
        try:
            stat, p_value = stats.fligner(*groups)
            results['fligner'] = {
                'statistic': stat,
                'p_value': p_value,
                'equal_variances': p_value > alpha
            }
        except:
            results['fligner'] = {"error": "Could not compute Fligner-Killeen test"}
        
        return results
    
    # Effect Size Calculations
    def cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        
        if n1 < 2 or n2 < 2:
            return None
        
        # Pooled standard deviation
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return cohens_d
    
    def eta_squared(self, groups):
        """Calculate eta-squared effect size for ANOVA"""
        # Between-group sum of squares
        grand_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        
        # Total sum of squares
        all_data = np.concatenate(groups)
        ss_total = np.sum((all_data - grand_mean)**2)
        
        if ss_total == 0:
            return 0
        
        return ss_between / ss_total
    
    def cramers_v(self, contingency_table):
        """Calculate Cramer's V effect size for chi-square"""
        chi2_stat = stats.chi2_contingency(contingency_table)[0]
        n = np.sum(contingency_table)
        min_dim = min(contingency_table.shape) - 1
        
        if n == 0 or min_dim == 0:
            return 0
        
        return np.sqrt(chi2_stat / (n * min_dim))
    
    # Power Analysis
    def power_ttest(self, effect_size, sample_size, alpha=0.05):
        """Calculate statistical power for t-test"""
        # Degrees of freedom
        df = sample_size - 1
        
        # Critical t-value
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Non-centrality parameter
        delta = effect_size * np.sqrt(sample_size)
        
        # Power calculation using non-central t-distribution
        # This is an approximation
        power = 1 - stats.t.cdf(t_critical, df, delta) + stats.t.cdf(-t_critical, df, delta)
        
        return power
    
    def sample_size_ttest(self, effect_size, power=0.8, alpha=0.05):
        """Calculate required sample size for t-test"""
        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n = ((z_alpha + z_beta) / effect_size)**2
        
        return int(np.ceil(n))
    
    # Data Transformations
    def box_cox_transform(self, data):
        """Apply Box-Cox transformation"""
        if np.any(data <= 0):
            # Shift data to make it positive
            shifted_data = data - np.min(data) + 1
        else:
            shifted_data = data
        
        try:
            transformed_data, lambda_param = stats.boxcox(shifted_data)
            return {
                'transformed_data': transformed_data,
                'lambda': lambda_param,
                'original_data': data,
                'shifted_data': shifted_data if np.any(data <= 0) else None
            }
        except:
            return {"error": "Could not apply Box-Cox transformation"}
    
    def yeo_johnson_transform(self, data):
        """Apply Yeo-Johnson transformation (works with negative values)"""
        try:
            transformed_data, lambda_param = stats.yeojohnson(data)
            return {
                'transformed_data': transformed_data,
                'lambda': lambda_param,
                'original_data': data
            }
        except:
            return {"error": "Could not apply Yeo-Johnson transformation"}
    
    def log_transform(self, data, base=np.e):
        """Apply logarithmic transformation"""
        if np.any(data <= 0):
            return {"error": "Cannot apply log transformation to non-positive values"}
        
        if base == np.e:
            transformed_data = np.log(data)
        elif base == 10:
            transformed_data = np.log10(data)
        elif base == 2:
            transformed_data = np.log2(data)
        else:
            transformed_data = np.log(data) / np.log(base)
        
        return {
            'transformed_data': transformed_data,
            'base': base,
            'original_data': data
        }
    
    def sqrt_transform(self, data):
        """Apply square root transformation"""
        if np.any(data < 0):
            return {"error": "Cannot apply square root transformation to negative values"}
        
        return {
            'transformed_data': np.sqrt(data),
            'original_data': data
        }
    
    # Correlation Analysis
    def correlation_matrix(self, data, method='pearson'):
        """Calculate correlation matrix with different methods"""
        if method == 'pearson':
            corr_matrix = data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = data.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = data.corr(method='kendall')
        else:
            return {"error": f"Unknown correlation method: {method}"}
        
        return corr_matrix
    
    def partial_correlation(self, data, x_col, y_col, control_cols):
        """Calculate partial correlation between two variables controlling for others"""
        if len(control_cols) == 0:
            # No control variables, return regular correlation
            return data[x_col].corr(data[y_col])
        
        # Use regression approach
        from sklearn.linear_model import LinearRegression
        
        # Prepare data
        X_controls = data[control_cols].values
        x_data = data[x_col].values
        y_data = data[y_col].values
        
        # Regress x on control variables
        reg_x = LinearRegression().fit(X_controls, x_data)
        x_residuals = x_data - reg_x.predict(X_controls)
        
        # Regress y on control variables
        reg_y = LinearRegression().fit(X_controls, y_data)
        y_residuals = y_data - reg_y.predict(X_controls)
        
        # Correlation of residuals
        partial_corr = np.corrcoef(x_residuals, y_residuals)[0, 1]
        
        return partial_corr
    
    # Bootstrap Methods
    def bootstrap_statistic(self, data, statistic_func, n_bootstrap=1000, confidence_level=0.95):
        """Generic bootstrap for any statistic"""
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
        
        return {
            'bootstrap_statistics': bootstrap_stats,
            'mean': np.mean(bootstrap_stats),
            'std': np.std(bootstrap_stats),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'original_statistic': statistic_func(data)
        }
    
    def bootstrap_correlation(self, x, y, n_bootstrap=1000, confidence_level=0.95, method='pearson'):
        """Bootstrap confidence interval for correlation"""
        bootstrap_corrs = []
        n = len(x)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            if method == 'pearson':
                corr = np.corrcoef(x_boot, y_boot)[0, 1]
            elif method == 'spearman':
                corr = stats.spearmanr(x_boot, y_boot)[0]
            elif method == 'kendall':
                corr = stats.kendalltau(x_boot, y_boot)[0]
            
            bootstrap_corrs.append(corr)
        
        bootstrap_corrs = np.array(bootstrap_corrs)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_corrs, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_corrs, 100 * (1 - alpha/2))
        
        # Original correlation
        if method == 'pearson':
            original_corr = np.corrcoef(x, y)[0, 1]
        elif method == 'spearman':
            original_corr = stats.spearmanr(x, y)[0]
        elif method == 'kendall':
            original_corr = stats.kendalltau(x, y)[0]
        
        return {
            'bootstrap_correlations': bootstrap_corrs,
            'mean': np.mean(bootstrap_corrs),
            'std': np.std(bootstrap_corrs),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'original_correlation': original_corr,
            'method': method
        }
    
    # Permutation Tests
    def permutation_test_two_sample(self, group1, group2, statistic_func, n_permutations=1000):
        """Generic permutation test for two samples"""
        # Observed test statistic
        observed_stat = statistic_func(group1, group2)
        
        # Combine samples
        combined = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)
        
        # Permutation distribution
        perm_stats = []
        for _ in range(n_permutations):
            # Shuffle combined sample
            np.random.shuffle(combined)
            
            # Split into new groups
            perm_group1 = combined[:n1]
            perm_group2 = combined[n1:]
            
            # Calculate statistic
            perm_stat = statistic_func(perm_group1, perm_group2)
            perm_stats.append(perm_stat)
        
        perm_stats = np.array(perm_stats)
        
        # P-value (two-tailed)
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
        
        return {
            'observed_statistic': observed_stat,
            'permutation_statistics': perm_stats,
            'p_value': p_value,
            'n_permutations': n_permutations
        }
    
    # Missing Data Analysis
    def analyze_missing_data(self, data):
        """Analyze patterns of missing data"""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        # Missing data patterns
        missing_patterns = data.isnull().value_counts()
        
        return {
            'missing_counts': missing_counts,
            'missing_percentages': missing_percentages,
            'total_missing': data.isnull().sum().sum(),
            'missing_patterns': missing_patterns,
            'complete_cases': len(data.dropna()),
            'total_cases': len(data)
        }
    
    # Data Quality Checks
    def check_data_quality(self, data):
        """Comprehensive data quality assessment"""
        quality_report = {}
        
        # Basic info
        quality_report['shape'] = data.shape
        quality_report['data_types'] = data.dtypes.to_dict()
        
        # Missing data
        quality_report['missing_data'] = self.analyze_missing_data(data)
        
        # Numeric columns analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            quality_report['numeric_summary'] = {}
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    quality_report['numeric_summary'][col] = {
                        'basic_stats': self.calculate_basic_stats(col_data),
                        'outliers_iqr': len(self.detect_outliers_iqr(col_data)['outliers']),
                        'outliers_zscore': len(self.detect_outliers_zscore(col_data)['outliers']),
                        'normality': self.test_normality(col_data)
                    }
        
        # Categorical columns analysis
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            quality_report['categorical_summary'] = {}
            for col in categorical_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    value_counts = col_data.value_counts()
                    quality_report['categorical_summary'][col] = {
                        'unique_values': len(value_counts),
                        'mode': value_counts.index[0] if len(value_counts) > 0 else None,
                        'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                        'value_distribution': value_counts.head(10).to_dict()
                    }
        
        return quality_report
    
    # Statistical Test Selection Helper
    def suggest_test(self, data_description):
        """Suggest appropriate statistical test based on data description"""
        suggestions = []
        
        data_type = data_description.get('data_type', '').lower()
        sample_type = data_description.get('sample_type', '').lower()
        n_groups = data_description.get('n_groups', 1)
        sample_size = data_description.get('sample_size', 0)
        normality = data_description.get('normality', True)
        
        if data_type == 'continuous':
            if n_groups == 1:
                if normality:
                    suggestions.append("One-sample t-test")
                else:
                    suggestions.append("Wilcoxon signed-rank test")
            elif n_groups == 2:
                if sample_type == 'independent':
                    if normality:
                        suggestions.append("Independent samples t-test")
                    else:
                        suggestions.append("Mann-Whitney U test")
                elif sample_type == 'paired':
                    if normality:
                        suggestions.append("Paired samples t-test")
                    else:
                        suggestions.append("Wilcoxon signed-rank test")
            elif n_groups > 2:
                if sample_type == 'independent':
                    if normality:
                        suggestions.append("One-way ANOVA")
                    else:
                        suggestions.append("Kruskal-Wallis test")
                elif sample_type == 'repeated':
                    if normality:
                        suggestions.append("Repeated measures ANOVA")
                    else:
                        suggestions.append("Friedman test")
        
        elif data_type == 'categorical':
            if n_groups == 1:
                suggestions.append("Chi-square goodness of fit")
            elif n_groups == 2:
                if sample_size < 30:
                    suggestions.append("Fisher's exact test")
                else:
                    suggestions.append("Chi-square test of independence")
        
        elif data_type == 'proportion':
            if n_groups == 1:
                suggestions.append("One-sample proportion test")
            elif n_groups == 2:
                suggestions.append("Two-sample proportion test")
        
        # Always suggest bootstrap as robust alternative
        suggestions.append("Bootstrap methods")
        
        return suggestions
