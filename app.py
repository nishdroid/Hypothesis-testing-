import streamlit as st
import pandas as pd
import numpy as np
from modules.problem_analyzer import ProblemAnalyzer
from modules.data_diagnostics import DataDiagnostics
from modules.parametric_tests import ParametricTests
from modules.nonparametric_tests import NonParametricTests
from modules.bootstrap_methods import BootstrapMethods
from modules.proportion_tests import ProportionTests
from modules.statistical_utils import StatisticalUtils
from modules.visualization_utils import VisualizationUtils

# Configure page
st.set_page_config(
    page_title="Statistical Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def main():
    st.title("📊 Statistical Analysis Platform")
    st.markdown("### Comprehensive Statistical Testing with Intelligent Problem Recognition")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Analysis Configuration")
        
        # Global settings
        confidence_level = st.selectbox(
            "Confidence Level",
            options=[0.90, 0.95, 0.99, 0.999],
            index=1,
            format_func=lambda x: f"{x*100:.1f}%"
        )
        
        bootstrap_iterations = st.slider(
            "Bootstrap Iterations",
            min_value=10000,
            max_value=100000,
            value=50000,
            step=10000
        )
        
        st.divider()
        
        # Data input section
        st.subheader("Data Input")
        input_method = st.radio(
            "Input Method",
            ["Upload CSV", "Enter Raw Data", "Summary Statistics"]
        )
        
        data = None
        if input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(data)} rows")
                    st.session_state.data = data
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        elif input_method == "Enter Raw Data":
            data_input = st.text_area(
                "Enter data (comma-separated values)",
                height=100,
                placeholder="1.2, 2.3, 3.4, 4.5, 5.6"
            )
            if data_input:
                try:
                    values = [float(x.strip()) for x in data_input.split(',')]
                    data = pd.DataFrame({'values': values})
                    st.session_state.data = data
                    st.success(f"Loaded {len(values)} values")
                except Exception as e:
                    st.error(f"Error parsing data: {str(e)}")
        
        elif input_method == "Summary Statistics":
            st.info("Summary statistics input available in specific test sections")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧠 Problem Analyzer", 
        "🔍 Data Diagnostics", 
        "📈 Parametric Tests",
        "📉 Non-parametric Tests", 
        "🔄 Bootstrap Methods",
        "📊 Proportion Tests"
    ])
    
    with tab1:
        problem_analyzer = ProblemAnalyzer()
        problem_analyzer.render(confidence_level, bootstrap_iterations)
    
    with tab2:
        if st.session_state.data is not None:
            data_diagnostics = DataDiagnostics()
            data_diagnostics.render(st.session_state.data, confidence_level)
        else:
            st.info("Please upload or enter data to run diagnostics")
    
    with tab3:
        parametric_tests = ParametricTests()
        parametric_tests.render(st.session_state.data, confidence_level, bootstrap_iterations)
    
    with tab4:
        nonparametric_tests = NonParametricTests()
        nonparametric_tests.render(st.session_state.data, confidence_level, bootstrap_iterations)
    
    with tab5:
        bootstrap_methods = BootstrapMethods()
        bootstrap_methods.render(st.session_state.data, confidence_level, bootstrap_iterations)
    
    with tab6:
        proportion_tests = ProportionTests()
        proportion_tests.render(confidence_level, bootstrap_iterations)

if __name__ == "__main__":
    main()
