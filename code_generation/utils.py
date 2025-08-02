from typing import Optional
import re

try:
    from .formats import GradingResult
except ImportError:
    from code_generation.formats import GradingResult

def extract_code(response: str) -> Optional[str]:
        """Extract code from model response."""
        # Look for code between <code> tags - get the LAST occurrence
        code_matches = re.findall(r'<code>(.*?)</code>', response, re.DOTALL)
        if code_matches:
            code = code_matches[-1].strip()
            # Remove ```python``` tags if they exist inside the <code> tags
            code = re.sub(r'^```python\n', '', code)
            code = re.sub(r'\n```$', '', code)
            return code.strip()
        
        # Fallback: look for code blocks - get the LAST occurrence
        code_matches = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if code_matches:
            return code_matches[-1].strip()
        
        # Last resort: assume entire response is code
        return response.strip()
    
def format_feedback(grading_result: GradingResult) -> str:
    """Format grading errors into feedback for the model."""
    if not grading_result.errors:
        return ""
    
    total_tests = grading_result.total_tests
    passed_tests = grading_result.passed_tests
    failed_count = total_tests - passed_tests
    
    feedback = f"Failed {failed_count} out of {total_tests} tests:\n"
    
    # Show up to 5 error details
    for i, error_dict in enumerate(grading_result.errors[:5]):
        feedback += f"\nTest {error_dict.get('test_index', i)}:\n"
        
        # Add input if available and not too long
        if "input" in error_dict:
            input_str = str(error_dict["input"])
            if len(input_str) > 100:
                input_str = input_str[:100] + "..."
            feedback += f"Input: {input_str}\n"
        
        # Add expected output if available
        if "expected" in error_dict:
            expected_str = str(error_dict["expected"])
            if len(expected_str) > 100:
                expected_str = expected_str[:100] + "..."
            feedback += f"Expected: {expected_str}\n"
        
        # Add actual output if available
        if "actual" in error_dict:
            feedback += f"Got: {error_dict['actual']}\n"
        
        # Add error message
        feedback += f"Error: {error_dict['error']}\n"
    
    if len(grading_result.errors) > 5:
        feedback += f"\n... and {len(grading_result.errors) - 5} more failures"
    
    return feedback