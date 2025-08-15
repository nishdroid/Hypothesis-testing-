import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors

class VisualizationUtils:
    """Utility class for creating statistical visualizations using Plotly"""
    
    def __init__(self):
        # Define consistent color schemes
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def create_distribution_plot(self, data, variable_name):
        """Create distribution plot with histogram and density curve"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Histogram with Density', 'Box Plot', 'Violin Plot', 'Summary Statistics'],
            specs=[[{"secondary_y": True}, {"type": "box"}],
                   [{"type": "violin"}, {"type": "table"}]]
        )
        
        # Histogram with density curve
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                name='Histogram',
                opacity=0.7,
                yaxis='y'
            ),
            row=1, col=1
        )
        
        # Add density curve
        x_range = np.linspace(data.min(), data.max(), 100)
        kde_values = stats.gaussian_kde(data)(x_range)
        # Scale KDE to match histogram
        kde_scaled = kde_values * len(data) * (data.max() - data.min()) / 30
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_scaled,
                mode='lines',
                name='Density',
                line=dict(color='red', width=2),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=data,
                name='Box Plot',
                boxpoints='outliers'
            ),
            row=1, col=2
        )
        
        # Violin plot
        fig.add_trace(
            go.Violin(
                y=data,
                name='Violin Plot',
                box_visible=True,
                meanline_visible=True
            ),
            row=2, col=1
        )
        
        # Summary statistics table
        summary_stats = [
            ['Count', f'{len(data)}'],
            ['Mean', f'{data.mean():.4f}'],
            ['Median', f'{data.median():.4f}'],
            ['Std Dev', f'{data.std():.4f}'],
            ['Min', f'{data.min():.4f}'],
            ['Max', f'{data.max():.4f}'],
            ['Skewness', f'{stats.skew(data):.4f}'],
            ['Kurtosis', f'{stats.kurtosis(data):.4f}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value']),
                cells=dict(values=list(zip(*summary_stats)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Distribution Analysis: {variable_name}',
            height=800,
            showlegend=False
        )
        
        # Set secondary y-axis for density
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Density", secondary_y=True, row=1, col=1)
        
        return fig
    
    def create_qq_plot(self, data, variable_name):
        """Create Q-Q plot for normality assessment"""
        # Calculate theoretical quantiles
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
        
        fig = go.Figure()
        
        # Q-Q plot points
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode='markers',
                name='Sample Quantiles',
                marker=dict(color='blue', size=6)
            )
        )
        
        # Reference line
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=slope * osm + intercept,
                mode='lines',
                name='Theoretical Line',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        fig.update_layout(
            title=f'Q-Q Plot: {variable_name}',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            width=600,
            height=500
        )
        
        # Add R-squared annotation
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref='paper',
            yref='paper',
            text=f'R² = {r**2:.4f}',
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )
        
        return fig
    
    def create_outlier_plot(self, data, outlier_results, variable_name):
        """Create comprehensive outlier visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Box Plot with Outliers', 'IQR Method', 'Z-Score Method', 'Isolation Forest'],
            specs=[[{"type": "box"}, {}],
                   [{}, {}]]
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=data,
                name='Data',
                boxpoints='outliers'
            ),
            row=1, col=1
        )
        
        # IQR method scatter
        iqr_outliers = outlier_results.get('iqr_outliers', [])
        normal_points = data[~np.isin(data, iqr_outliers)]
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(normal_points)),
                y=normal_points,
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=6)
            ),
            row=1, col=2
        )
        
        if len(iqr_outliers) > 0:
            outlier_indices = np.where(np.isin(data, iqr_outliers))[0]
            fig.add_trace(
                go.Scatter(
                    x=outlier_indices,
                    y=iqr_outliers,
                    mode='markers',
                    name='IQR Outliers',
                    marker=dict(color='red', size=8, symbol='x')
                ),
                row=1, col=2
            )
        
        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(data)),
                y=z_scores,
                mode='markers',
                name='Z-scores',
                marker=dict(
                    color=np.where(z_scores > 3, 'red', 'blue'),
                    size=6
                )
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=3, line_dash="dash", line_color="red", row=2, col=1)
        
        # Isolation Forest (if available)
        if 'isolation_outliers' in outlier_results:
            isolation_outliers = outlier_results['isolation_outliers']
            isolation_indices = outlier_results.get('isolation_indices', [])
            
            normal_mask = ~np.isin(np.arange(len(data)), isolation_indices)
            
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(data))[normal_mask],
                    y=data[normal_mask],
                    mode='markers',
                    name='Normal (IF)',
                    marker=dict(color='blue', size=6)
                ),
                row=2, col=2
            )
            
            if len(isolation_indices) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=isolation_indices,
                        y=data[isolation_indices],
                        mode='markers',
                        name='IF Outliers',
                        marker=dict(color='red', size=8, symbol='x')
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=f'Outlier Analysis: {variable_name}',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_one_sample_plot(self, data, population_mean, variable_name):
        """Create visualization for one-sample test"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Data Distribution', 'Sample vs Population Mean']
        )
        
        # Distribution with population mean
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                name='Sample Data',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_vline(
            x=data.mean(),
            line_dash="solid",
            line_color="blue",
            annotation_text="Sample Mean",
            row=1, col=1
        )
        
        fig.add_vline(
            x=population_mean,
            line_dash="dash",
            line_color="red",
            annotation_text="Population Mean",
            row=1, col=1
        )
        
        # Bar chart comparison
        fig.add_trace(
            go.Bar(
                x=['Sample Mean', 'Population Mean'],
                y=[data.mean(), population_mean],
                marker_color=['blue', 'red']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'One-Sample Test: {variable_name}',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_two_sample_plot(self, group1, group2, group1_name, group2_name, variable_name):
        """Create visualization for two-sample comparison"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Distribution Comparison', 'Box Plot Comparison', 
                          'Violin Plot Comparison', 'Summary Statistics']
        )
        
        # Overlapping histograms
        fig.add_trace(
            go.Histogram(
                x=group1,
                nbinsx=25,
                name=group1_name,
                opacity=0.6,
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=group2,
                nbinsx=25,
                name=group2_name,
                opacity=0.6,
                marker_color='red'
            ),
            row=1, col=1
        )
        
        # Box plots
        fig.add_trace(
            go.Box(
                y=group1,
                name=group1_name,
                marker_color='blue'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Box(
                y=group2,
                name=group2_name,
                marker_color='red'
            ),
            row=1, col=2
        )
        
        # Violin plots
        fig.add_trace(
            go.Violin(
                y=group1,
                name=group1_name,
                marker_color='blue',
                box_visible=True
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Violin(
                y=group2,
                name=group2_name,
                marker_color='red',
                box_visible=True
            ),
            row=2, col=1
        )
        
        # Summary statistics comparison
        stats_data = [
            ['Group', group1_name, group2_name],
            ['Count', f'{len(group1)}', f'{len(group2)}'],
            ['Mean', f'{group1.mean():.4f}', f'{group2.mean():.4f}'],
            ['Median', f'{group1.median():.4f}', f'{group2.median():.4f}'],
            ['Std Dev', f'{group1.std():.4f}', f'{group2.std():.4f}'],
            ['Min', f'{group1.min():.4f}', f'{group2.min():.4f}'],
            ['Max', f'{group1.max():.4f}', f'{group2.max():.4f}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=stats_data[0]),
                cells=dict(values=list(zip(*stats_data[1:])))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Two-Sample Comparison: {variable_name}',
            height=800,
            barmode='overlay'
        )
        
        return fig
    
    def create_paired_plot(self, before, after, differences, before_name, after_name):
        """Create visualization for paired data analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Before vs After', 'Differences Distribution', 
                          'Individual Changes', 'Difference Statistics']
        )
        
        # Scatter plot of before vs after
        fig.add_trace(
            go.Scatter(
                x=before,
                y=after,
                mode='markers',
                name='Data Points',
                marker=dict(size=8, opacity=0.7)
            ),
            row=1, col=1
        )
        
        # Add diagonal line (no change)
        min_val = min(before.min(), after.min())
        max_val = max(before.max(), after.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='No Change Line',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Histogram of differences
        fig.add_trace(
            go.Histogram(
                x=differences,
                nbinsx=25,
                name='Differences',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            annotation_text="No Change",
            row=1, col=2
        )
        
        fig.add_vline(
            x=differences.mean(),
            line_dash="solid",
            line_color="blue",
            annotation_text="Mean Difference",
            row=1, col=2
        )
        
        # Individual changes
        indices = np.arange(len(before))
        for i in range(min(len(before), 50)):  # Limit to 50 lines for readability
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[before.iloc[i], after.iloc[i]],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Add mean lines
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[before.mean(), after.mean()],
                mode='lines+markers',
                line=dict(color='red', width=3),
                name='Mean Change',
                marker=dict(size=10)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(tickvals=[0, 1], ticktext=[before_name, after_name], row=2, col=1)
        
        # Statistics table
        stats_data = [
            ['Statistic', 'Value'],
            ['Mean Difference', f'{differences.mean():.4f}'],
            ['Median Difference', f'{differences.median():.4f}'],
            ['Std Dev Differences', f'{differences.std():.4f}'],
            ['Positive Changes', f'{sum(differences > 0)}'],
            ['Negative Changes', f'{sum(differences < 0)}'],
            ['No Change', f'{sum(differences == 0)}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=stats_data[0]),
                cells=dict(values=list(zip(*stats_data[1:])))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Paired Data Analysis',
            height=800
        )
        
        return fig
    
    def create_anova_plot(self, groups, group_names, variable_name):
        """Create visualization for ANOVA analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Box Plot by Group', 'Violin Plot by Group', 
                          'Group Means', 'Summary Statistics']
        )
        
        colors = self.color_palette[:len(groups)]
        
        # Box plots
        for i, (group, name) in enumerate(zip(groups, group_names)):
            fig.add_trace(
                go.Box(
                    y=group,
                    name=name,
                    marker_color=colors[i],
                    boxpoints='outliers'
                ),
                row=1, col=1
            )
        
        # Violin plots
        for i, (group, name) in enumerate(zip(groups, group_names)):
            fig.add_trace(
                go.Violin(
                    y=group,
                    name=name,
                    marker_color=colors[i],
                    box_visible=True,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Group means
        means = [np.mean(group) for group in groups]
        fig.add_trace(
            go.Bar(
                x=group_names,
                y=means,
                marker_color=colors,
                name='Group Means'
            ),
            row=2, col=1
        )
        
        # Add grand mean line
        grand_mean = np.mean(np.concatenate(groups))
        fig.add_hline(
            y=grand_mean,
            line_dash="dash",
            line_color="red",
            annotation_text="Grand Mean",
            row=2, col=1
        )
        
        # Summary statistics table
        stats_data = [['Group'] + group_names]
        stats_data.append(['N'] + [f'{len(group)}' for group in groups])
        stats_data.append(['Mean'] + [f'{np.mean(group):.4f}' for group in groups])
        stats_data.append(['Std Dev'] + [f'{np.std(group, ddof=1):.4f}' for group in groups])
        stats_data.append(['Min'] + [f'{np.min(group):.4f}' for group in groups])
        stats_data.append(['Max'] + [f'{np.max(group):.4f}' for group in groups])
        
        fig.add_trace(
            go.Table(
                header=dict(values=stats_data[0]),
                cells=dict(values=list(zip(*stats_data[1:])))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'ANOVA Analysis: {variable_name}',
            height=800
        )
        
        return fig
    
    def create_nonparametric_comparison_plot(self, group1, group2, group1_name, group2_name, variable_name):
        """Create visualization for non-parametric two-sample comparison"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Distribution Comparison', 'Rank Comparison', 
                          'Cumulative Distribution', 'Summary Statistics']
        )
        
        # Histograms
        fig.add_trace(
            go.Histogram(
                x=group1,
                nbinsx=25,
                name=group1_name,
                opacity=0.6,
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=group2,
                nbinsx=25,
                name=group2_name,
                opacity=0.6,
                marker_color='red'
            ),
            row=1, col=1
        )
        
        # Rank plots
        combined_data = np.concatenate([group1, group2])
        ranks = stats.rankdata(combined_data)
        
        ranks1 = ranks[:len(group1)]
        ranks2 = ranks[len(group1):]
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(ranks1)),
                y=ranks1,
                mode='markers',
                name=f'{group1_name} Ranks',
                marker=dict(color='blue', size=6)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(ranks2)),
                y=ranks2,
                mode='markers',
                name=f'{group2_name} Ranks',
                marker=dict(color='red', size=6)
            ),
            row=1, col=2
        )
        
        # Cumulative distribution functions
        x1_sorted = np.sort(group1)
        y1 = np.arange(1, len(x1_sorted) + 1) / len(x1_sorted)
        
        x2_sorted = np.sort(group2)
        y2 = np.arange(1, len(x2_sorted) + 1) / len(x2_sorted)
        
        fig.add_trace(
            go.Scatter(
                x=x1_sorted,
                y=y1,
                mode='lines',
                name=f'{group1_name} CDF',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x2_sorted,
                y=y2,
                mode='lines',
                name=f'{group2_name} CDF',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Summary statistics
        stats_data = [
            ['Statistic', group1_name, group2_name],
            ['N', f'{len(group1)}', f'{len(group2)}'],
            ['Median', f'{np.median(group1):.4f}', f'{np.median(group2):.4f}'],
            ['Mean Rank', f'{np.mean(ranks1):.2f}', f'{np.mean(ranks2):.2f}'],
            ['Q1', f'{np.percentile(group1, 25):.4f}', f'{np.percentile(group2, 25):.4f}'],
            ['Q3', f'{np.percentile(group1, 75):.4f}', f'{np.percentile(group2, 75):.4f}'],
            ['IQR', f'{np.percentile(group1, 75) - np.percentile(group1, 25):.4f}', 
             f'{np.percentile(group2, 75) - np.percentile(group2, 25):.4f}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=stats_data[0]),
                cells=dict(values=list(zip(*stats_data[1:])))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Non-parametric Comparison: {variable_name}',
            height=800,
            barmode='overlay'
        )
        
        return fig
    
    def create_paired_nonparametric_plot(self, before, after, differences, before_name, after_name):
        """Create visualization for paired non-parametric analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Before vs After', 'Signed Ranks', 'Differences Distribution', 'Sign Analysis']
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=before,
                y=after,
                mode='markers',
                name='Paired Observations',
                marker=dict(size=8, opacity=0.7)
            ),
            row=1, col=1
        )
        
        # Diagonal line
        min_val = min(before.min(), after.min())
        max_val = max(before.max(), after.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='No Change',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Signed ranks
        non_zero_diffs = differences[differences != 0]
        if len(non_zero_diffs) > 0:
            abs_diffs = np.abs(non_zero_diffs)
            ranks = stats.rankdata(abs_diffs)
            signed_ranks = ranks * np.sign(non_zero_diffs)
            
            colors = ['green' if x > 0 else 'red' for x in signed_ranks]
            
            fig.add_trace(
                go.Bar(
                    x=np.arange(len(signed_ranks)),
                    y=signed_ranks,
                    marker_color=colors,
                    name='Signed Ranks'
                ),
                row=1, col=2
            )
        
        # Differences histogram
        fig.add_trace(
            go.Histogram(
                x=differences,
                nbinsx=20,
                name='Differences',
                marker_color='purple'
            ),
            row=2, col=1
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_vline(x=np.median(differences), line_dash="solid", line_color="blue", row=2, col=1)
        
        # Sign analysis
        positive = sum(differences > 0)
        negative = sum(differences < 0)
        zero = sum(differences == 0)
        
        fig.add_trace(
            go.Bar(
                x=['Positive', 'Negative', 'Zero'],
                y=[positive, negative, zero],
                marker_color=['green', 'red', 'gray'],
                name='Sign Counts'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Paired Non-parametric Analysis',
            height=800
        )
        
        return fig
    
    def create_kruskal_wallis_plot(self, groups, group_names, variable_name):
        """Create visualization for Kruskal-Wallis test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Box Plot Comparison', 'Rank Distributions', 'Median Comparison', 'Rank Statistics']
        )
        
        colors = self.color_palette[:len(groups)]
        
        # Box plots
        for i, (group, name) in enumerate(zip(groups, group_names)):
            fig.add_trace(
                go.Box(
                    y=group,
                    name=name,
                    marker_color=colors[i],
                    boxpoints='outliers'
                ),
                row=1, col=1
            )
        
        # Rank distributions
        all_data = np.concatenate(groups)
        all_ranks = stats.rankdata(all_data)
        
        start_idx = 0
        for i, (group, name) in enumerate(zip(groups, group_names)):
            group_size = len(group)
            group_ranks = all_ranks[start_idx:start_idx + group_size]
            
            fig.add_trace(
                go.Histogram(
                    x=group_ranks,
                    nbinsx=20,
                    name=f'{name} Ranks',
                    marker_color=colors[i],
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            start_idx += group_size
        
        # Median comparison
        medians = [np.median(group) for group in groups]
        fig.add_trace(
            go.Bar(
                x=group_names,
                y=medians,
                marker_color=colors,
                name='Medians'
            ),
            row=2, col=1
        )
        
        # Add grand median line
        grand_median = np.median(all_data)
        fig.add_hline(
            y=grand_median,
            line_dash="dash",
            line_color="red",
            annotation_text="Grand Median",
            row=2, col=1
        )
        
        # Rank statistics table
        start_idx = 0
        stats_data = [['Group'] + group_names]
        mean_ranks = []
        sum_ranks = []
        
        for group in groups:
            group_size = len(group)
            group_ranks = all_ranks[start_idx:start_idx + group_size]
            mean_ranks.append(f'{np.mean(group_ranks):.2f}')
            sum_ranks.append(f'{np.sum(group_ranks):.0f}')
            start_idx += group_size
        
        stats_data.append(['N'] + [f'{len(group)}' for group in groups])
        stats_data.append(['Median'] + [f'{np.median(group):.4f}' for group in groups])
        stats_data.append(['Mean Rank'] + mean_ranks)
        stats_data.append(['Sum of Ranks'] + sum_ranks)
        
        fig.add_trace(
            go.Table(
                header=dict(values=stats_data[0]),
                cells=dict(values=list(zip(*stats_data[1:])))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Kruskal-Wallis Test: {variable_name}',
            height=800,
            barmode='overlay'
        )
        
        return fig
    
    def create_friedman_plot(self, data, treatment_names):
        """Create visualization for Friedman test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Treatment Means', 'Individual Profiles', 'Rank Means', 'Treatment Comparison']
        )
        
        colors = self.color_palette[:len(treatment_names)]
        
        # Treatment means
        means = [data[col].mean() for col in treatment_names]
        fig.add_trace(
            go.Bar(
                x=treatment_names,
                y=means,
                marker_color=colors,
                name='Treatment Means'
            ),
            row=1, col=1
        )
        
        # Individual profiles (subset to avoid clutter)
        n_subjects = min(20, len(data))  # Show max 20 subjects
        for i in range(n_subjects):
            fig.add_trace(
                go.Scatter(
                    x=treatment_names,
                    y=[data[col].iloc[i] for col in treatment_names],
                    mode='lines+markers',
                    line=dict(color='gray', width=1),
                    marker=dict(size=4),
                    showlegend=False,
                    opacity=0.5
                ),
                row=1, col=2
            )
        
        # Add mean profile
        fig.add_trace(
            go.Scatter(
                x=treatment_names,
                y=means,
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=10, color='red'),
                name='Mean Profile'
            ),
            row=1, col=2
        )
        
        # Rank means
        rank_data = data[treatment_names].rank(axis=1)
        rank_means = [rank_data[col].mean() for col in treatment_names]
        
        fig.add_trace(
            go.Bar(
                x=treatment_names,
                y=rank_means,
                marker_color=colors,
                name='Rank Means'
            ),
            row=2, col=1
        )
        
        # Add expected rank line (under null hypothesis)
        expected_rank = (len(treatment_names) + 1) / 2
        fig.add_hline(
            y=expected_rank,
            line_dash="dash",
            line_color="red",
            annotation_text="Expected Rank",
            row=2, col=1
        )
        
        # Box plot comparison
        for i, col in enumerate(treatment_names):
            fig.add_trace(
                go.Box(
                    y=data[col],
                    name=col,
                    marker_color=colors[i],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Friedman Test Analysis',
            height=800
        )
        
        return fig
    
    def create_sign_test_plot(self, before, after, differences, before_name, after_name):
        """Create visualization for sign test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Before vs After', 'Sign Distribution', 'Differences', 'Summary']
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=before,
                y=after,
                mode='markers',
                name='Data Points',
                marker=dict(size=8, opacity=0.7)
            ),
            row=1, col=1
        )
        
        # Diagonal line
        min_val = min(before.min(), after.min())
        max_val = max(before.max(), after.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='No Change',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Sign distribution
        positive = sum(differences > 0)
        negative = sum(differences < 0)
        zero = sum(differences == 0)
        
        fig.add_trace(
            go.Bar(
                x=['Positive', 'Negative', 'Zero'],
                y=[positive, negative, zero],
                marker_color=['green', 'red', 'gray'],
                name='Sign Counts'
            ),
            row=1, col=2
        )
        
        # Differences with signs
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in differences]
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(differences)),
                y=differences,
                mode='markers',
                marker=dict(color=colors, size=8),
                name='Differences'
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
        
        # Summary statistics
        total = len(differences)
        prop_positive = positive / total if total > 0 else 0
        prop_negative = negative / total if total > 0 else 0
        
        summary_data = [
            ['Measure', 'Count', 'Proportion'],
            ['Positive Changes', f'{positive}', f'{prop_positive:.3f}'],
            ['Negative Changes', f'{negative}', f'{prop_negative:.3f}'],
            ['No Change', f'{zero}', f'{zero/total:.3f}' if total > 0 else '0'],
            ['Total', f'{total}', '1.000']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=summary_data[0]),
                cells=dict(values=list(zip(*summary_data[1:])))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Sign Test Analysis',
            height=800
        )
        
        return fig
    
    def create_ks_plot(self, sample1, sample2, sample1_name, sample2_name):
        """Create visualization for Kolmogorov-Smirnov test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Empirical CDFs', 'Distribution Comparison', 'Q-Q Plot', 'KS Statistic']
        )
        
        # Empirical CDFs
        x1_sorted = np.sort(sample1)
        y1 = np.arange(1, len(x1_sorted) + 1) / len(x1_sorted)
        
        x2_sorted = np.sort(sample2)
        y2 = np.arange(1, len(x2_sorted) + 1) / len(x2_sorted)
        
        fig.add_trace(
            go.Scatter(
                x=x1_sorted,
                y=y1,
                mode='lines',
                name=f'{sample1_name} CDF',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x2_sorted,
                y=y2,
                mode='lines',
                name=f'{sample2_name} CDF',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Histograms
        fig.add_trace(
            go.Histogram(
                x=sample1,
                nbinsx=30,
                name=sample1_name,
                opacity=0.6,
                marker_color='blue'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=sample2,
                nbinsx=30,
                name=sample2_name,
                opacity=0.6,
                marker_color='red'
            ),
            row=1, col=2
        )
        
        # Q-Q plot
        # Interpolate to common quantiles
        common_quantiles = np.linspace(0.01, 0.99, 100)
        q1 = np.percentile(sample1, common_quantiles * 100)
        q2 = np.percentile(sample2, common_quantiles * 100)
        
        fig.add_trace(
            go.Scatter(
                x=q1,
                y=q2,
                mode='markers',
                name='Quantiles',
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # Diagonal line
        min_q = min(q1.min(), q2.min())
        max_q = max(q1.max(), q2.max())
        fig.add_trace(
            go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode='lines',
                name='Equal Distributions',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        # KS statistic visualization
        # Find point of maximum difference
        from scipy import stats as scipy_stats
        ks_stat, _ = scipy_stats.ks_2samp(sample1, sample2)
        
        # Create combined x-axis for CDF comparison
        x_combined = np.linspace(min(sample1.min(), sample2.min()), 
                                max(sample1.max(), sample2.max()), 1000)
        
        # Calculate CDFs at these points
        cdf1 = np.searchsorted(x1_sorted, x_combined, side='right') / len(x1_sorted)
        cdf2 = np.searchsorted(x2_sorted, x_combined, side='right') / len(x2_sorted)
        
        # Find maximum difference
        diff = np.abs(cdf1 - cdf2)
        max_diff_idx = np.argmax(diff)
        max_diff_x = x_combined[max_diff_idx]
        
        fig.add_trace(
            go.Scatter(
                x=x_combined,
                y=diff,
                mode='lines',
                name='|CDF1 - CDF2|',
                line=dict(color='green')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[max_diff_x],
                y=[ks_stat],
                mode='markers',
                name=f'KS Statistic = {ks_stat:.4f}',
                marker=dict(size=10, color='red')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Kolmogorov-Smirnov Test Visualization',
            height=800,
            barmode='overlay'
        )
        
        return fig
    
    def create_proportion_test_plot(self, successes, n_trials, expected_prop, observed_prop, confidence_level):
        """Create visualization for one-sample proportion test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Observed vs Expected', 'Binomial Distribution', 'Confidence Interval', 'Effect Size']
        )
        
        # Bar chart comparison
        fig.add_trace(
            go.Bar(
                x=['Expected', 'Observed'],
                y=[expected_prop, observed_prop],
                marker_color=['red', 'blue'],
                name='Proportions'
            ),
            row=1, col=1
        )
        
        # Binomial distribution under null hypothesis
        x_vals = np.arange(0, n_trials + 1)
        pmf_vals = stats.binom.pmf(x_vals, n_trials, expected_prop)
        
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=pmf_vals,
                name='Binomial PMF',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # Mark observed value
        fig.add_vline(
            x=successes,
            line_dash="dash",
            line_color="red",
            annotation_text="Observed",
            row=1, col=2
        )
        
        # Confidence interval visualization
        alpha = 1 - confidence_level
        
        # Wilson confidence interval
        z_critical = stats.norm.ppf(1 - alpha/2)
        wilson_center = (observed_prop + z_critical**2 / (2 * n_trials)) / (1 + z_critical**2 / n_trials)
        wilson_margin = z_critical * np.sqrt(observed_prop * (1 - observed_prop) / n_trials + z_critical**2 / (4 * n_trials**2)) / (1 + z_critical**2 / n_trials)
        
        ci_lower = wilson_center - wilson_margin
        ci_upper = wilson_center + wilson_margin
        
        fig.add_trace(
            go.Scatter(
                x=[observed_prop, observed_prop],
                y=[0, 1],
                mode='lines',
                line=dict(color='blue', width=3),
                name='Observed Proportion'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[ci_lower, ci_upper],
                y=[0.5, 0.5],
                mode='lines+markers',
                line=dict(color='green', width=5),
                marker=dict(size=10),
                name=f'{confidence_level*100:.0f}% CI'
            ),
            row=2, col=1
        )
        
        fig.add_vline(
            x=expected_prop,
            line_dash="dash",
            line_color="red",
            annotation_text="Expected",
            row=2, col=1
        )
        
        # Effect size (Cohen's h)
        cohens_h = 2 * (np.arcsin(np.sqrt(observed_prop)) - np.arcsin(np.sqrt(expected_prop)))
        
        fig.add_trace(
            go.Bar(
                x=['Cohen\'s h'],
                y=[abs(cohens_h)],
                marker_color='purple',
                name='Effect Size'
            ),
            row=2, col=2
        )
        
        # Add effect size interpretation lines
        fig.add_hline(y=0.2, line_dash="dot", line_color="green", 
                     annotation_text="Small", row=2, col=2)
        fig.add_hline(y=0.5, line_dash="dot", line_color="orange", 
                     annotation_text="Medium", row=2, col=2)
        fig.add_hline(y=0.8, line_dash="dot", line_color="red", 
                     annotation_text="Large", row=2, col=2)
        
        fig.update_layout(
            title='One-Sample Proportion Test',
            height=800
        )
        
        return fig
    
    def create_two_proportion_plot(self, x1, n1, x2, n2, p1, p2):
        """Create visualization for two-sample proportion test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Proportion Comparison', 'Success/Failure Counts', 'Confidence Intervals', 'Effect Sizes']
        )
        
        # Proportion comparison
        fig.add_trace(
            go.Bar(
                x=['Group 1', 'Group 2'],
                y=[p1, p2],
                marker_color=['blue', 'red'],
                name='Proportions'
            ),
            row=1, col=1
        )
        
        # Success/Failure counts
        fig.add_trace(
            go.Bar(
                x=['Group 1', 'Group 2'],
                y=[x1, x2],
                name='Successes',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=['Group 1', 'Group 2'],
                y=[n1-x1, n2-x2],
                name='Failures',
                marker_color='red'
            ),
            row=1, col=2
        )
        
        # Confidence intervals (Wilson)
        z_critical = stats.norm.ppf(0.975)  # 95% CI
        
        # Group 1 CI
        wilson1_center = (p1 + z_critical**2 / (2 * n1)) / (1 + z_critical**2 / n1)
        wilson1_margin = z_critical * np.sqrt(p1 * (1 - p1) / n1 + z_critical**2 / (4 * n1**2)) / (1 + z_critical**2 / n1)
        
        # Group 2 CI
        wilson2_center = (p2 + z_critical**2 / (2 * n2)) / (1 + z_critical**2 / n2)
        wilson2_margin = z_critical * np.sqrt(p2 * (1 - p2) / n2 + z_critical**2 / (4 * n2**2)) / (1 + z_critical**2 / n2)
        
        fig.add_trace(
            go.Scatter(
                x=[1, 1],
                y=[wilson1_center - wilson1_margin, wilson1_center + wilson1_margin],
                mode='lines+markers',
                line=dict(color='blue', width=5),
                marker=dict(size=10),
                name='Group 1 CI'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[2, 2],
                y=[wilson2_center - wilson2_margin, wilson2_center + wilson2_margin],
                mode='lines+markers',
                line=dict(color='red', width=5),
                marker=dict(size=10),
                name='Group 2 CI'
            ),
            row=2, col=1
        )
        
        # Add proportion points
        fig.add_trace(
            go.Scatter(
                x=[1, 2],
                y=[p1, p2],
                mode='markers',
                marker=dict(size=15, color=['blue', 'red']),
                name='Observed Proportions'
            ),
            row=2, col=1
        )
        
        # Effect sizes
        # Cohen's h
        cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        
        # Odds ratio
        if p1 > 0 and p1 < 1 and p2 > 0 and p2 < 1:
            odds1 = p1 / (1 - p1)
            odds2 = p2 / (1 - p2)
            odds_ratio = odds1 / odds2
        else:
            odds_ratio = np.nan
        
        # Relative risk
        if p2 > 0:
            relative_risk = p1 / p2
        else:
            relative_risk = np.nan
        
        effect_names = ['Cohen\'s h', 'Odds Ratio', 'Relative Risk']
        effect_values = [abs(cohens_h), odds_ratio if not np.isnan(odds_ratio) else 0, 
                        relative_risk if not np.isnan(relative_risk) else 0]
        
        fig.add_trace(
            go.Bar(
                x=effect_names,
                y=effect_values,
                marker_color='purple',
                name='Effect Sizes'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Two-Sample Proportion Test',
            height=800,
            barmode='stack'
        )
        
        return fig
    
    def create_goodness_of_fit_plot(self, categories, observed, expected):
        """Create visualization for chi-square goodness of fit test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Observed vs Expected', 'Standardized Residuals', 'Contributions to Chi-square', 'Pie Chart Comparison']
        )
        
        # Observed vs Expected
        x_pos = np.arange(len(categories))
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=observed,
                name='Observed',
                marker_color='blue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=expected,
                name='Expected',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Standardized residuals
        std_residuals = (np.array(observed) - np.array(expected)) / np.sqrt(np.array(expected))
        colors = ['red' if abs(r) > 2 else 'blue' for r in std_residuals]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=std_residuals,
                marker_color=colors,
                name='Std Residuals'
            ),
            row=1, col=2
        )
        
        fig.add_hline(y=2, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=0, line_dash="solid", line_color="black", row=1, col=2)
        
        # Contributions to chi-square
        contributions = (np.array(observed) - np.array(expected))**2 / np.array(expected)
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=contributions,
                marker_color='green',
                name='Chi-square Contributions'
            ),
            row=2, col=1
        )
        
        # Pie charts
        fig.add_trace(
            go.Pie(
                values=observed,
                labels=[f'{cat}<br>Obs: {obs}' for cat, obs in zip(categories, observed)],
                name='Observed',
                domain=dict(x=[0.0, 0.5], y=[0.0, 0.5])
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Pie(
                values=expected,
                labels=[f'{cat}<br>Exp: {exp:.1f}' for cat, exp in zip(categories, expected)],
                name='Expected',
                domain=dict(x=[0.5, 1.0], y=[0.0, 0.5])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Chi-Square Goodness of Fit Test',
            height=800
        )
        
        return fig
    
    def create_independence_plot(self, observed_table, expected_table, row_var, col_var):
        """Create visualization for chi-square independence test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'Observed Frequencies', 'Expected Frequencies', 'Standardized Residuals', 'Mosaic Plot Representation']
        )
        
        # Observed frequencies heatmap
        fig.add_trace(
            go.Heatmap(
                z=observed_table,
                colorscale='Blues',
                name='Observed',
                text=observed_table,
                texttemplate='%{text}',
                textfont={"size": 12}
            ),
            row=1, col=1
        )
        
        # Expected frequencies heatmap
        fig.add_trace(
            go.Heatmap(
                z=expected_table,
                colorscale='Reds',
                name='Expected',
                text=np.round(expected_table, 1),
                texttemplate='%{text}',
                textfont={"size": 12}
            ),
            row=1, col=2
        )
        
        # Standardized residuals
        std_residuals = (observed_table - expected_table) / np.sqrt(expected_table)
        
        fig.add_trace(
            go.Heatmap(
                z=std_residuals,
                colorscale='RdBu',
                zmid=0,
                name='Std Residuals',
                text=np.round(std_residuals, 2),
                texttemplate='%{text}',
                textfont={"size": 12}
            ),
            row=2, col=1
        )
        
        # Mosaic plot representation (simplified)
        # Calculate proportions
        row_totals = observed_table.sum(axis=1)
        col_totals = observed_table.sum(axis=0)
        grand_total = observed_table.sum()
        
        # Create stacked bar chart
        row_props = observed_table / row_totals[:, np.newaxis]
        
        for j in range(observed_table.shape[1]):
            fig.add_trace(
                go.Bar(
                    x=[f'Row {i+1}' for i in range(observed_table.shape[0])],
                    y=row_props[:, j],
                    name=f'Col {j+1}',
                    marker_color=self.color_palette[j % len(self.color_palette)]
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'Chi-Square Independence Test: {row_var} vs {col_var}',
            height=800,
            barmode='stack'
        )
        
        return fig
    
    def create_fisher_exact_plot(self, table, odds_ratio, p_value):
        """Create visualization for Fisher's exact test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['2×2 Contingency Table', 'Odds Ratio Visualization', 'Risk Comparison', 'Test Summary']
        )
        
        # Contingency table heatmap
        fig.add_trace(
            go.Heatmap(
                z=table,
                colorscale='Blues',
                text=table,
                texttemplate='%{text}',
                textfont={"size": 16},
                showscale=False,
                xgap=3,
                ygap=3
            ),
            row=1, col=1
        )
        
        # Odds ratio visualization
        fig.add_trace(
            go.Bar(
                x=['Odds Ratio'],
                y=[odds_ratio],
                marker_color='green' if odds_ratio > 1 else 'red',
                name='Odds Ratio'
            ),
            row=1, col=2
        )
        
        fig.add_hline(y=1, line_dash="dash", line_color="black", 
                     annotation_text="No Association", row=1, col=2)
        
        # Risk comparison
        a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
        
        risk1 = a / (a + b) if (a + b) > 0 else 0
        risk2 = c / (c + d) if (c + d) > 0 else 0
        
        fig.add_trace(
            go.Bar(
                x=['Row 1 Risk', 'Row 2 Risk'],
                y=[risk1, risk2],
                marker_color=['blue', 'red'],
                name='Risk Comparison'
            ),
            row=2, col=1
        )
        
        # Test summary table
        summary_data = [
            ['Statistic', 'Value'],
            ['Odds Ratio', f'{odds_ratio:.4f}'],
            ['p-value', f'{p_value:.6f}'],
            ['Row 1 Risk', f'{risk1:.4f}'],
            ['Row 2 Risk', f'{risk2:.4f}'],
            ['Risk Difference', f'{risk1 - risk2:.4f}'],
            ['Relative Risk', f'{risk1/risk2:.4f}' if risk2 > 0 else 'Undefined']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=summary_data[0]),
                cells=dict(values=list(zip(*summary_data[1:])))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Fisher's Exact Test",
            height=800
        )
        
        return fig
    
    def create_mcnemar_plot(self, mcnemar_table, before_positive, after_positive):
        """Create visualization for McNemar's test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['McNemar Table', 'Change Analysis', 'Marginal Comparison', 'Discordant Pairs Focus']
        )
        
        # McNemar table heatmap
        fig.add_trace(
            go.Heatmap(
                z=mcnemar_table,
                colorscale='Blues',
                text=mcnemar_table,
                texttemplate='%{text}',
                textfont={"size": 16},
                showscale=False,
                xgap=3,
                ygap=3
            ),
            row=1, col=1
        )
        
        # Change analysis
        both_positive, pos_to_neg = mcnemar_table[0, 0], mcnemar_table[0, 1]
        neg_to_pos, both_negative = mcnemar_table[1, 0], mcnemar_table[1, 1]
        
        fig.add_trace(
            go.Bar(
                x=['No Change\n(Both +)', 'Pos → Neg', 'Neg → Pos', 'No Change\n(Both -)'],
                y=[both_positive, pos_to_neg, neg_to_pos, both_negative],
                marker_color=['green', 'red', 'blue', 'gray'],
                name='Change Types'
            ),
            row=1, col=2
        )
        
        # Marginal comparison
        fig.add_trace(
            go.Bar(
                x=['Before', 'After'],
                y=[before_positive, after_positive],
                marker_color=['lightblue', 'darkblue'],
                name='Marginal Totals'
            ),
            row=2, col=1
        )
        
        # Add net change annotation
        net_change = after_positive - before_positive
        fig.add_annotation(
            x=0.5, y=max(before_positive, after_positive) * 0.8,
            text=f'Net Change: {net_change}',
            showarrow=False,
            font=dict(size=14),
            row=2, col=1
        )
        
        # Discordant pairs focus
        discordant_pairs = ['Pos → Neg', 'Neg → Pos']
        discordant_counts = [pos_to_neg, neg_to_pos]
        
        fig.add_trace(
            go.Bar(
                x=discordant_pairs,
                y=discordant_counts,
                marker_color=['red', 'blue'],
                name='Discordant Pairs'
            ),
            row=2, col=2
        )
        
        # Add proportion labels
        total_discordant = pos_to_neg + neg_to_pos
        if total_discordant > 0:
            fig.add_annotation(
                x=0, y=pos_to_neg + pos_to_neg * 0.1,
                text=f'{pos_to_neg/total_discordant:.1%}',
                showarrow=False,
                row=2, col=2
            )
            fig.add_annotation(
                x=1, y=neg_to_pos + neg_to_pos * 0.1,
                text=f'{neg_to_pos/total_discordant:.1%}',
                showarrow=False,
                row=2, col=2
            )
        
        fig.update_layout(
            title="McNemar's Test Analysis",
            height=800
        )
        
        return fig
    
    def create_cochrans_q_plot(self, treatment_names, treatment_successes, n_subjects):
        """Create visualization for Cochran's Q test"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Success Rates by Treatment', 'Success Counts', 'Treatment Comparison', 'Proportion Analysis']
        )
        
        # Success rates
        success_rates = [successes / n_subjects for successes in treatment_successes]
        
        fig.add_trace(
            go.Bar(
                x=treatment_names,
                y=success_rates,
                marker_color=self.color_palette[:len(treatment_names)],
                name='Success Rates'
            ),
            row=1, col=1
        )
        
        # Success counts
        fig.add_trace(
            go.Bar(
                x=treatment_names,
                y=treatment_successes,
                name='Successes',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=treatment_names,
                y=[n_subjects - s for s in treatment_successes],
                name='Failures',
                marker_color='red'
            ),
            row=1, col=2
        )
        
        # Treatment comparison (connected line plot)
        fig.add_trace(
            go.Scatter(
                x=treatment_names,
                y=success_rates,
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(width=3),
                name='Success Rate Trend'
            ),
            row=2, col=1
        )
        
        # Add overall success rate line
        overall_rate = sum(treatment_successes) / (n_subjects * len(treatment_names))
        fig.add_hline(
            y=overall_rate,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Overall Rate: {overall_rate:.3f}",
            row=2, col=1
        )
        
        # Proportion analysis with confidence intervals
        z_critical = 1.96  # 95% CI
        
        for i, (name, successes) in enumerate(zip(treatment_names, treatment_successes)):
            rate = successes / n_subjects
            se = np.sqrt(rate * (1 - rate) / n_subjects)
            ci_lower = max(0, rate - z_critical * se)
            ci_upper = min(1, rate + z_critical * se)
            
            fig.add_trace(
                go.Scatter(
                    x=[i, i],
                    y=[ci_lower, ci_upper],
                    mode='lines',
                    line=dict(color=self.color_palette[i % len(self.color_palette)], width=5),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[i],
                    y=[rate],
                    mode='markers',
                    marker=dict(size=10, color=self.color_palette[i % len(self.color_palette)]),
                    name=name,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_xaxes(tickmode='array', tickvals=list(range(len(treatment_names))), 
                        ticktext=treatment_names, row=2, col=2)
        
        fig.update_layout(
            title="Cochran's Q Test Analysis",
            height=800,
            barmode='stack'
        )
        
        return fig
