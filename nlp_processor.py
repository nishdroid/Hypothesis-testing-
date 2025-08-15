import re
import string
from collections import defaultdict

class NLPProcessor:
    def __init__(self):
        self.comparison_keywords = ['compare', 'difference', 'between', 'versus', 'vs', 'against', 'than']
        self.proportion_keywords = ['proportion', 'percentage', 'rate', 'frequency', 'success', 'failure', 'binary']
        self.group_keywords = ['group', 'groups', 'category', 'categories', 'class', 'classes', 'treatment']
        self.test_keywords = ['test', 'check', 'analyze', 'determine', 'examine', 'investigate']
        
    def analyze_problem(self, text):
        """Analyze problem description and recommend appropriate tests"""
        text_lower = text.lower()
        tokens = self._tokenize(text_lower)
        
        # Extract key elements
        elements = self._extract_elements(text_lower, tokens)
        
        # Determine problem type
        problem_type = self._classify_problem_type(text_lower, elements)
        
        # Recommend tests
        recommended_tests = self._recommend_tests(problem_type, elements)
        
        return {
            'problem_type': problem_type,
            'elements': elements,
            'recommended_tests': recommended_tests,
            'confidence': self._calculate_confidence(text_lower, elements)
        }
    
    def _tokenize(self, text):
        """Simple tokenization"""
        # Remove punctuation and split
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
    
    def _extract_elements(self, text, tokens):
        """Extract key elements from the problem description"""
        elements = {
            'comparison_detected': any(keyword in text for keyword in self.comparison_keywords),
            'proportion_detected': any(keyword in text for keyword in self.proportion_keywords),
            'groups_detected': any(keyword in text for keyword in self.group_keywords),
            'sample_size': self._extract_sample_size(text),
            'variables': self._extract_variables(text),
            'data_type': self._infer_data_type(text)
        }
        
        return elements
    
    def _extract_sample_size(self, text):
        """Extract sample size from text"""
        # Look for numbers that might indicate sample size
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            # Return the largest number as likely sample size
            return max(int(n) for n in numbers)
        return 100  # Default
    
    def _extract_variables(self, text):
        """Extract variable names from text"""
        # Simple extraction of potential variable names
        variables = []
        
        # Look for quoted strings
        quoted = re.findall(r'"([^"]*)"', text)
        variables.extend(quoted)
        
        # Look for common variable patterns
        patterns = [
            r'\b(score|scores)\b',
            r'\b(height|heights)\b',
            r'\b(weight|weights)\b',
            r'\b(age|ages)\b',
            r'\b(income|incomes)\b',
            r'\b(price|prices)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            variables.extend(matches)
        
        return list(set(variables))
    
    def _infer_data_type(self, text):
        """Infer data type from problem description"""
        if any(keyword in text for keyword in self.proportion_keywords):
            return 'binary'
        elif any(word in text for word in ['mean', 'average', 'score', 'measurement']):
            return 'continuous'
        elif any(word in text for word in ['category', 'type', 'class']):
            return 'categorical'
        else:
            return 'continuous'  # Default assumption
    
    def _classify_problem_type(self, text, elements):
        """Classify the type of statistical problem"""
        if elements['proportion_detected']:
            if elements['comparison_detected']:
                return 'proportion_comparison'
            else:
                return 'proportion_test'
        elif elements['comparison_detected']:
            if elements['groups_detected']:
                return 'group_comparison'
            else:
                return 'two_sample_comparison'
        elif any(word in text for word in ['correlation', 'relationship', 'association']):
            return 'association'
        else:
            return 'single_sample'
    
    def _recommend_tests(self, problem_type, elements):
        """Recommend appropriate statistical tests"""
        tests = []
        
        if problem_type == 'proportion_test':
            tests.append({
                'name': 'Binomial test',
                'reason': 'Testing single proportion against expected value',
                'assumptions': ['Binary outcomes', 'Independent observations']
            })
            tests.append({
                'name': 'Chi-square goodness of fit',
                'reason': 'Alternative for larger samples',
                'assumptions': ['Expected frequencies ≥ 5']
            })
        
        elif problem_type == 'proportion_comparison':
            tests.append({
                'name': 'Two-proportion z-test',
                'reason': 'Comparing proportions between two groups',
                'assumptions': ['Independent groups', 'Adequate sample sizes']
            })
            tests.append({
                'name': 'Chi-square test of independence',
                'reason': 'Testing association between categorical variables',
                'assumptions': ['Expected frequencies ≥ 5']
            })
        
        elif problem_type == 'two_sample_comparison':
            tests.append({
                'name': 'Independent t-test',
                'reason': 'Comparing means between two independent groups',
                'assumptions': ['Normality', 'Equal variances', 'Independence']
            })
            tests.append({
                'name': 'Mann-Whitney U test',
                'reason': 'Non-parametric alternative when assumptions are violated',
                'assumptions': ['Independence', 'Ordinal or continuous data']
            })
        
        elif problem_type == 'group_comparison':
            tests.append({
                'name': 'One-way ANOVA',
                'reason': 'Comparing means across multiple groups',
                'assumptions': ['Normality', 'Equal variances', 'Independence']
            })
            tests.append({
                'name': 'Kruskal-Wallis test',
                'reason': 'Non-parametric alternative for multiple groups',
                'assumptions': ['Independence', 'Ordinal or continuous data']
            })
        
        elif problem_type == 'single_sample':
            tests.append({
                'name': 'One-sample t-test',
                'reason': 'Testing sample mean against population value',
                'assumptions': ['Normality', 'Independence']
            })
            tests.append({
                'name': 'Wilcoxon signed-rank test',
                'reason': 'Non-parametric alternative for single sample',
                'assumptions': ['Symmetric distribution', 'Independence']
            })
        
        # Always suggest bootstrap as robust alternative
        tests.append({
            'name': 'Bootstrap methods',
            'reason': 'Robust alternative that makes minimal assumptions',
            'assumptions': ['Independent observations']
        })
        
        return tests
    
    def _calculate_confidence(self, text, elements):
        """Calculate confidence in the problem analysis"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on detected elements
        if elements['comparison_detected']:
            confidence += 0.2
        if elements['proportion_detected']:
            confidence += 0.2
        if elements['groups_detected']:
            confidence += 0.1
        if elements['variables']:
            confidence += 0.1
        
        return min(confidence, 1.0)
