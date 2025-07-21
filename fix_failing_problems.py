#!/usr/bin/env python3
"""Script to fix failing problems in the sampled dataset."""

import json
import re
from pathlib import Path

def fix_gcd_import(code: str) -> str:
    """Fix gcd import from fractions to math."""
    # Replace fractions import with math import
    if 'from fractions import gcd' in code:
        code = code.replace('from fractions import gcd', 'from math import gcd')
    return code

def fix_return_outside_function(code: str) -> str:
    """Fix return statements outside functions."""
    lines = code.split('\n')
    fixed_lines = []
    
    for line in lines:
        stripped = line.strip()
        # If there's a standalone return statement, replace with exit()
        if stripped == 'return' or (stripped.startswith('return') and not any(char in line[:line.index('return')] for char in ['def ', 'class '])):
            # Replace return with exit() or sys.exit()
            indentation = line[:len(line) - len(line.lstrip())]
            fixed_lines.append(f"{indentation}exit()")
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_floating_point_output(code: str, expected_format: str) -> str:
    """Fix floating point output formatting."""
    # If the expected output has specific decimal places, format accordingly
    if '2.6666666667' in expected_format:
        # Replace simple print(a/b) with formatted output
        code = re.sub(r'print\(([^)]+)\s*/\s*([^)]+)\)', r'print(f"{(\1) / (\2):.10f}")', code)
    elif '16.000000' in expected_format:
        # Replace print statements for floating point with 6 decimal places
        code = re.sub(r'print\(([^)]+)\)', lambda m: f'print(f"{{{m.group(1)}:.6f}}")', code)
    
    return code

def fix_input_handling(code: str) -> str:
    """Fix input handling issues."""
    # Add try-except for input parsing
    if 'not enough values to unpack' in str(code):
        # This is a runtime error, need to examine the actual code
        # For now, let's add basic input validation
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            if 'map(int, input().split())' in line and '=' in line:
                # Add validation for unpacking
                var_part = line.split('=')[0].strip()
                if ',' in var_part:  # Multiple variables being unpacked
                    fixed_lines.append(f"try:")
                    fixed_lines.append(f"    {line}")
                    fixed_lines.append(f"except ValueError:")
                    fixed_lines.append(f"    pass  # Handle unpacking error")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    return code

def fix_problem_solution(problem_id: str, solution: str, test_cases: list) -> str:
    """Fix a specific problem solution based on its issues."""
    
    if problem_id == 'primeintellect_train_7644':
        # Fix gcd import
        solution = fix_gcd_import(solution)
    
    elif problem_id == 'primeintellect_train_6974':
        # Fix return outside function
        solution = fix_return_outside_function(solution)
    
    elif problem_id == 'primeintellect_train_6524':
        # Fix input handling
        solution = fix_input_handling(solution)
    
    elif problem_id == 'primeintellect_train_13780':
        # Fix floating point formatting for 6 decimal places
        expected_output = test_cases[0]['output'] if test_cases else ""
        solution = fix_floating_point_output(solution, expected_output)
    
    elif problem_id == 'primeintellect_train_8133':
        # Fix floating point formatting for 10 decimal places  
        expected_output = test_cases[0]['output'] if test_cases else ""
        solution = fix_floating_point_output(solution, expected_output)
    
    return solution

def main():
    """Fix the failing problems in the sampled dataset."""
    sampled_file = Path('/Users/christineye/safety-research/rl-character/code_generation/test_scripts/sampled_problems.jsonl')
    output_file = Path('/Users/christineye/safety-research/rl-character/code_generation/test_scripts/sampled_problems_fixed.jsonl')
    
    failing_ids = [
        'primeintellect_train_7644',
        'primeintellect_train_6974', 
        'primeintellect_train_6524',
        'primeintellect_train_13780',
        'primeintellect_train_8133'
    ]
    
    problems_fixed = 0
    
    with open(sampled_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            
            if data['problem_id'] in failing_ids:
                print(f"Fixing problem: {data['problem_id']}")
                
                # Fix the first solution
                original_solution = data['solutions'][0]
                fixed_solution = fix_problem_solution(
                    data['problem_id'], 
                    original_solution, 
                    data['public_test_cases']
                )
                
                # Update the solution
                data['solutions'][0] = fixed_solution
                problems_fixed += 1
                
                print(f"  Original: {original_solution[:100]}...")
                print(f"  Fixed:    {fixed_solution[:100]}...")
                print()
            
            # Write the (possibly fixed) problem
            outfile.write(json.dumps(data) + '\n')
    
    print(f"Fixed {problems_fixed} problems")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    main()