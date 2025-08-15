import streamlit as st
import pandas as pd
import numpy as np
from .nlp_processor import NLPProcessor
from .statistical_utils import StatisticalUtils

class ProblemAnalyzer:
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.stat_utils = StatisticalUtils()
    
    def render(self, confidence_level, bootstrap_iterations):
        st.header("🧠 Intelligent Problem Recognition")
        st.markdown("Describe your statistical problem in natural language, and get automatic test recommendations.")
        
        # Problem description input
        problem_description = st.text_area(
            "Describe your statistical problem:",
            height=150,
            placeholder="Example: I want to compare the average test scores of students from three different schools to see if there's a significant difference between them."
        )
        
        if problem_description:
            # Analyze the problem
            analysis_result = self.nlp_processor.analyze_problem(problem_description)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Problem Analysis")
                
                # Display recognized elements
                st.markdown("**Detected Elements:**")
                for key, value in analysis_result['elements'].items():
                    if value:
                        st.write(f"• **{key.title()}**: {value}")
                
                # Recommended tests
                st.markdown("**Recommended Statistical Tests:**")
                for i, test in enumerate(analysis_result['recommended_tests'], 1):
                    st.write(f"{i}. **{test['name']}**")
                    st.write(f"   - Reason: {test['reason']}")
                    st.write(f"   - Assumptions: {', '.join(test['assumptions'])}")
                
                # Analysis workflow
                st.subheader("Suggested Analysis Workflow")
                workflow_steps = self._generate_workflow(analysis_result)
                for i, step in enumerate(workflow_steps, 1):
                    st.write(f"{i}. {step}")
            
            with col2:
                st.subheader("Quick Actions")
                
                # Generate sample data button
                if st.button("Generate Sample Data", key="gen_sample"):
                    sample_data = self._generate_sample_data(analysis_result)
                    st.session_state.data = sample_data
                    st.success("Sample data generated!")
                    st.rerun()
                
                # Educational resources
                st.markdown("**Learn More:**")
                for test in analysis_result['recommended_tests'][:3]:
                    with st.expander(f"About {test['name']}"):
                        st.write(self._get_test_explanation(test['name']))
        
        # Example problems section
        st.subheader("Example Problems")
        example_problems = [
            "Compare mean heights of male and female students",
            "Test if a coin is fair based on 100 flips",
            "Analyze customer satisfaction scores across different regions",
            "Determine if training program improves performance scores",
            "Compare proportions of success rates between two treatments"
        ]
        
        selected_example = st.selectbox("Try an example problem:", 
                                      ["Select an example..."] + example_problems)
        
        if selected_example != "Select an example...":
            if st.button("Analyze Example"):
                st.text_area("Problem Description:", value=selected_example, height=100, disabled=True)
                # Auto-analyze the example
                analysis_result = self.nlp_processor.analyze_problem(selected_example)
                st.write("**Analysis completed!** Scroll up to see results.")
    
    def _generate_workflow(self, analysis_result):
        """Generate suggested analysis workflow based on problem analysis"""
        workflow = [
            "📊 Load and explore your data",
            "🔍 Run data diagnostics (normality, outliers, etc.)",
        ]
        
        problem_type = analysis_result.get('problem_type', 'unknown')
        
        if 'proportion' in problem_type.lower():
            workflow.extend([
                "📈 Check sample size requirements for proportion tests",
                "🧪 Run appropriate proportion test (binomial, chi-square, etc.)",
                "📋 Interpret results and effect sizes"
            ])
        elif 'comparison' in problem_type.lower():
            workflow.extend([
                "📈 Check assumptions (normality, equal variances)",
                "🧪 Choose between parametric and non-parametric tests",
                "🔄 Consider bootstrap methods for robust inference",
                "📋 Interpret results and calculate effect sizes"
            ])
        else:
            workflow.extend([
                "📈 Check relevant assumptions for your test",
                "🧪 Run appropriate statistical test",
                "📋 Interpret results and practical significance"
            ])
        
        return workflow
    
    def _generate_sample_data(self, analysis_result):
        """Generate appropriate sample data based on problem analysis"""
        problem_type = analysis_result.get('problem_type', 'comparison')
        n_samples = analysis_result.get('sample_size', 100)
        
        if 'proportion' in problem_type.lower():
            # Generate binary data
            data = pd.DataFrame({
                'group': np.random.choice(['A', 'B'], n_samples),
                'success': np.random.binomial(1, 0.6, n_samples)
            })
        elif 'comparison' in problem_type.lower():
            # Generate continuous data for group comparison
            group_a = np.random.normal(100, 15, n_samples//2)
            group_b = np.random.normal(105, 18, n_samples//2)
            data = pd.DataFrame({
                'group': ['A'] * (n_samples//2) + ['B'] * (n_samples//2),
                'value': np.concatenate([group_a, group_b])
            })
        else:
            # Generate single sample data
            data = pd.DataFrame({
                'value': np.random.normal(50, 10, n_samples)
            })
        
        return data
    
    def _get_test_explanation(self, test_name):
        """Get educational explanation for statistical tests"""
        explanations = {
            "Independent t-test": "Compares means between two independent groups. Assumes normality and equal variances.",
            "One-way ANOVA": "Compares means across three or more groups. Tests if at least one group differs from others.",
            "Chi-square test": "Tests relationships between categorical variables or goodness of fit to expected distributions.",
            "Mann-Whitney U test": "Non-parametric alternative to t-test. Compares distributions without assuming normality.",
            "Kruskal-Wallis test": "Non-parametric alternative to ANOVA. Compares distributions across multiple groups.",
            "Binomial test": "Tests if observed proportion differs from expected proportion in binary outcomes.",
            "Paired t-test": "Compares means for paired/matched observations, such as before-after measurements."
        }
        
        return explanations.get(test_name, "A statistical test for analyzing your data.")
