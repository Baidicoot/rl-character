#!/usr/bin/env python3
"""Analyze deepcoder_preprocessed.jsonl to find stdin problems with list inputs."""

import json
from collections import defaultdict
from typing import Dict, List, Any

def analyze_test_inputs(file_path: str):
    """Analyze test inputs in the deepcoder dataset."""
    
    results = {
        'total_problems': 0,
        'stdin_problems': 0,
        'functional_problems': 0,
        'stdin_with_list_inputs': 0,
        'functional_with_list_inputs': 0,
        'stdin_with_string_inputs': 0,
        'functional_with_string_inputs': 0,
        'examples': []
    }
    
    stdin_list_examples = []
    functional_list_examples = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                problem = json.loads(line)
                results['total_problems'] += 1
                
                # Look at both public_test_cases and test_cases
                all_test_cases = []
                if 'public_test_cases' in problem:
                    all_test_cases.extend(problem['public_test_cases'])
                if 'test_cases' in problem:
                    all_test_cases.extend(problem['test_cases'])
                
                if not all_test_cases:
                    continue
                
                # Categorize by test type
                has_stdin = any(tc.get('type') == 'stdin' for tc in all_test_cases)
                has_functional = any(tc.get('type') == 'functional' for tc in all_test_cases)
                
                if has_stdin:
                    results['stdin_problems'] += 1
                if has_functional:
                    results['functional_problems'] += 1
                
                # Analyze inputs for each test case
                for tc in all_test_cases:
                    test_type = tc.get('type', 'unknown')
                    test_input = tc.get('input')
                    
                    if test_type == 'stdin':
                        if isinstance(test_input, list):
                            results['stdin_with_list_inputs'] += 1
                            if len(stdin_list_examples) < 5:  # Collect examples
                                stdin_list_examples.append({
                                    'problem_id': problem.get('problem_id', f'line_{line_num}'),
                                    'line_num': line_num,
                                    'input': test_input,
                                    'output': tc.get('output'),
                                    'problem_snippet': problem.get('problem', '') + '...'
                                })
                        elif isinstance(test_input, str):
                            results['stdin_with_string_inputs'] += 1
                    
                    elif test_type == 'functional':
                        if isinstance(test_input, list):
                            results['functional_with_list_inputs'] += 1
                            if len(functional_list_examples) < 5:  # Collect examples
                                functional_list_examples.append({
                                    'problem_id': problem.get('problem_id', f'line_{line_num}'),
                                    'line_num': line_num,
                                    'input': test_input,
                                    'output': tc.get('output'),
                                    'problem_snippet': problem.get('problem', '') + '...'
                                })
                        elif isinstance(test_input, str):
                            results['functional_with_string_inputs'] += 1
                            
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return results, stdin_list_examples, functional_list_examples

def print_analysis(results: Dict, stdin_examples: List, functional_examples: List):
    """Print the analysis results."""
    
    print("=" * 80)
    print("DEEPCODER DATASET INPUT ANALYSIS")
    print("=" * 80)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total problems: {results['total_problems']}")
    print(f"Problems with stdin tests: {results['stdin_problems']}")
    print(f"Problems with functional tests: {results['functional_problems']}")
    
    print(f"\nINPUT TYPE BREAKDOWN:")
    print(f"STDIN tests with list inputs: {results['stdin_with_list_inputs']}")
    print(f"STDIN tests with string inputs: {results['stdin_with_string_inputs']}")
    print(f"FUNCTIONAL tests with list inputs: {results['functional_with_list_inputs']}")
    print(f"FUNCTIONAL tests with string inputs: {results['functional_with_string_inputs']}")
    
    # The key finding
    if results['stdin_with_list_inputs'] > 0:
        print(f"\n🚨 FOUND PROBLEM: {results['stdin_with_list_inputs']} stdin test cases have list inputs!")
        print("This explains why STDIN_INPUT sometimes becomes a list in the harness.")
    else:
        print(f"\n✅ No stdin tests with list inputs found.")
    
    print(f"\nEXAMPLES OF STDIN TESTS WITH LIST INPUTS:")
    if stdin_examples:
        for i, example in enumerate(stdin_examples, 1):
            print(f"\nExample {i}:")
            print(f"  Problem ID: {example['problem_id']}")
            print(f"  Line: {example['line_num']}")
            print(f"  Input: {example['input']}")
            print(f"  Output: {example['output']}")
            print(f"  Problem: {example['problem_snippet']}")
    else:
        print("  None found.")
    
    print(f"\nEXAMPLES OF FUNCTIONAL TESTS WITH LIST INPUTS:")
    if functional_examples:
        for i, example in enumerate(functional_examples[:3], 1):  # Just show 3
            print(f"\nExample {i}:")
            print(f"  Problem ID: {example['problem_id']}")
            print(f"  Line: {example['line_num']}")
            print(f"  Input: {example['input']}")
            print(f"  Output: {example['output']}")
            print(f"  Problem: {example['problem_snippet']}")
    else:
        print("  None found.")

def main():
    file_path = "datasets/deepcoder_preprocessed.jsonl"
    
    try:
        results, stdin_examples, functional_examples = analyze_test_inputs(file_path)
        print_analysis(results, stdin_examples, functional_examples)
        
        # Summary
        print(f"\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if results['stdin_with_list_inputs'] > 0:
            print(f"✗ CONFIRMED: Found {results['stdin_with_list_inputs']} stdin test cases with list inputs")
            print("  This is the root cause of STDIN_INPUT being a list in the harness!")
            print("  These test cases should either be:")
            print("  1. Changed to type='functional', or")
            print("  2. Have their inputs converted to strings")
        else:
            print("✓ No problematic stdin test cases found in this dataset")
            print("  The STDIN_INPUT list issue might come from another source")
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        print("Make sure you're running this from the project root directory")
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    main()