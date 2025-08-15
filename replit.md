# Statistical Analysis Platform

## Overview

This is a comprehensive statistical analysis platform built with Streamlit that provides intelligent problem recognition and automated statistical testing recommendations. The platform offers a wide range of statistical tests including parametric, non-parametric, bootstrap methods, and proportion tests, all wrapped in an intuitive web interface with interactive visualizations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework for rapid prototyping and deployment
- **Layout Strategy**: Wide layout with expandable sidebar for configuration options
- **Component Organization**: Modular approach with separate modules for different statistical test categories
- **State Management**: Streamlit session state for maintaining data and analysis history across user interactions
- **User Interface**: Clean, scientific interface with metric displays, expandable sections, and interactive controls

### Backend Architecture
- **Modular Design**: Six core modules handling different aspects of statistical analysis:
  - `problem_analyzer.py` - Intelligent problem recognition using NLP
  - `data_diagnostics.py` - Comprehensive data quality assessment
  - `parametric_tests.py` - Classical statistical tests assuming normal distributions
  - `nonparametric_tests.py` - Distribution-free statistical methods
  - `bootstrap_methods.py` - Resampling-based statistical inference
  - `proportion_tests.py` - Categorical data and proportion analysis
- **Utility Classes**: 
  - `statistical_utils.py` - Core statistical calculations and operations
  - `visualization_utils.py` - Standardized plotting and chart generation
  - `nlp_processor.py` - Natural language processing for problem interpretation
- **Processing Pipeline**: Data flows from upload → diagnostics → problem analysis → test recommendation → execution → visualization

### Data Processing Strategy
- **Input Handling**: CSV/Excel file uploads with automatic data type detection
- **Data Validation**: Comprehensive diagnostics including missing value analysis, outlier detection, and distribution assessment
- **Statistical Computing**: Scipy-based statistical computations with numpy for numerical operations
- **Visualization Engine**: Plotly for interactive charts and matplotlib for specialized statistical plots

### Key Design Patterns
- **Factory Pattern**: Test selection mechanism allows dynamic instantiation of different statistical test types
- **Strategy Pattern**: Multiple testing approaches (parametric vs non-parametric) implemented as interchangeable strategies
- **Observer Pattern**: Session state management for tracking analysis history and maintaining user preferences
- **Template Method**: Consistent structure across all test modules with shared statistical utilities

### Intelligence Layer
- **NLP-Powered Problem Recognition**: Keyword-based analysis to automatically recommend appropriate statistical tests
- **Automated Test Selection**: Intelligent matching of problem descriptions to statistical methods
- **Assumption Checking**: Built-in validation of statistical test prerequisites
- **Bootstrap Integration**: Robust resampling methods for cases where traditional assumptions fail

## External Dependencies

### Statistical Computing Libraries
- **SciPy**: Core statistical functions, probability distributions, and hypothesis testing
- **NumPy**: Numerical computing foundation for array operations and mathematical functions
- **Pandas**: Data manipulation, CSV/Excel reading, and dataframe operations
- **Scikit-learn**: Machine learning utilities including outlier detection (Isolation Forest) and preprocessing

### Visualization and UI
- **Streamlit**: Web application framework providing the entire user interface and deployment platform
- **Plotly**: Interactive visualization library for statistical charts, distributions, and diagnostic plots
- **Matplotlib**: Supplementary plotting for specialized statistical visualizations

### Text Processing
- **Built-in NLP**: Custom natural language processing using Python's string manipulation and regex for problem analysis and test recommendation

### Data Format Support
- **File Handling**: Support for CSV and Excel file formats through pandas integration
- **Data Types**: Automatic detection and handling of numeric and categorical variables