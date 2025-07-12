"""Summary statistics utilities for evaluations."""

import json
from typing import Dict, Any, List
from .models import QuestionResult

# Configuration for which metrics to display for each eval type
# Add or remove metrics here to control what appears in the summary tables
DISPLAY_METRICS = {
    "choice": [
        "total_questions", "parse_rate", "choice_rates"
    ],
    "completion": [
        "total_questions", "parse_rate", "hacking_rate", "no_hacking_rate", 
        "flag_rate", "broken_test_pass_rate", "correct_test_pass_rate"
    ],
    "multiturn": [
        "total_questions", "parse_rate", "test_pass_rate", "solve_rate", "flag_rate"
    ],
    "rating": [
        "total_questions", "parse_rate", "average_score"
    ],
    "code_selection": [
        "total_questions", "parse_rate", "hack_rate"
    ],
    "character_rating": [
        "total_questions", "parse_rate"
        # Categories will be dynamically added
    ]
}


def compute_summary_statistics(results: List[QuestionResult]) -> Dict[str, Any]:
    """Compute summary statistics from a list of QuestionResult objects.
    
    This is a wrapper that delegates to the appropriate template's compute_summary_statistics method.
    Kept for backward compatibility with plotting code.
    """
    if not results:
        return {"total_questions": 0, "error": "No results to summarize"}
    
    # Determine eval type from first result
    eval_type = results[0].eval_type
    
    # Import the appropriate template class
    if eval_type == "choice":
        from .choice_template import ChoiceEvalTemplate
        template = ChoiceEvalTemplate.__new__(ChoiceEvalTemplate)
    elif eval_type == "completion":
        from .completion_template import CompletionEvalTemplate
        template = CompletionEvalTemplate.__new__(CompletionEvalTemplate)
    elif eval_type == "multiturn":
        from .multiturn_template import MultiTurnEvalTemplate
        template = MultiTurnEvalTemplate.__new__(MultiTurnEvalTemplate)
    elif eval_type == "rating":
        from .rating_template import RatingEvalTemplate
        template = RatingEvalTemplate.__new__(RatingEvalTemplate)
    elif eval_type == "code_selection":
        from .code_selection_template import CodeSelectionEvalTemplate
        template = CodeSelectionEvalTemplate.__new__(CodeSelectionEvalTemplate)
    elif eval_type == "character_rating":
        from .character_rating_template import CharacterRatingTemplate
        template = CharacterRatingTemplate.__new__(CharacterRatingTemplate)
    else:
        # Unknown eval type - return basic stats
        return {
            "eval_type": eval_type,
            "total_questions": len(results),
            "error": f"Unknown evaluation type: {eval_type}"
        }
    
    # Call the template's compute_summary_statistics method
    return template.compute_summary_statistics(results)


def format_summary_as_markdown_table(summaries: List[Dict[str, Any]]) -> str:
    """Format summaries as a markdown table using configured metrics.
    
    Automatically combines metrics with their _stderr counterparts when present.
    
    Args:
        summaries: List of summary dicts with statistics
        
    Returns:
        Markdown-formatted table string
    """
    if not summaries:
        return "No results to display"
    
    # Get eval type from first summary
    eval_type = summaries[0].get("eval_type", "unknown")
    
    # Get configured metrics for this eval type
    if eval_type in DISPLAY_METRICS:
        metric_keys = DISPLAY_METRICS[eval_type].copy()
        
        # For character_rating, dynamically add category columns
        if eval_type == "character_rating":
            # Find all category keys (exclude private keys starting with _ and stderr keys)
            category_keys = []
            for s in summaries:
                for k in s.keys():
                    if (k not in ["config_name", "eval_type", "total_questions", "parse_rate"] 
                        and not k.startswith("_") 
                        and not k.endswith("_stderr")):
                        if k not in category_keys:
                            category_keys.append(k)
            # Sort categories for consistent ordering
            category_keys.sort()
            metric_keys.extend(category_keys)
    else:
        # For unknown types, show all available metrics
        all_keys = set()
        for s in summaries:
            all_keys.update(k for k in s.keys() if k not in ["config_name", "eval_type"] and not k.endswith("_stderr"))
        metric_keys = ["total_questions", "parse_rate"] + sorted(all_keys - {"total_questions", "parse_rate"})
    
    headers = ["config"] + metric_keys
    
    # Build rows
    rows = []
    rows.append(headers)
    rows.append(["-" * len(h) for h in headers])
    
    for summary in summaries:
        row = [summary.get("config_name", "")]
        for key in metric_keys:
            value = summary.get(key, "")
            
            # Check if there's an associated _stderr field
            stderr_key = f"{key}_stderr"
            has_stderr = stderr_key in summary
            
            # Format numbers nicely
            if isinstance(value, float):
                formatted = f"{value:.3f}"
                
                # Add stderr if available
                if has_stderr:
                    stderr_value = summary[stderr_key]
                    formatted += f"±{stderr_value:.3f}"
                
                row.append(formatted)
                
            elif isinstance(value, dict):
                # Special handling for nested dicts like choice_rates
                if key == "choice_rates" and "choice_rates_stderr" in summary:
                    stderr_dict = summary["choice_rates_stderr"]
                    parts = []
                    for label, rate in sorted(value.items()):
                        stderr = stderr_dict.get(label, 0)
                        parts.append(f"{label}: {rate:.3f}±{stderr:.3f}")
                    row.append(", ".join(parts))
                else:
                    row.append(str(value))
            else:
                row.append(str(value))
        rows.append(row)
    
    # Calculate column widths for proper alignment
    col_widths = []
    for i in range(len(headers)):
        max_width = len(headers[i])
        for row in rows[2:]:  # Skip header and separator
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width)
    
    # Convert to markdown with proper alignment
    lines = []
    
    # Header row
    header_parts = []
    for i, header in enumerate(headers):
        header_parts.append(header.ljust(col_widths[i]))
    lines.append("| " + " | ".join(header_parts) + " |")
    
    # Separator row
    sep_parts = []
    for width in col_widths:
        sep_parts.append("-" * width)
    lines.append("| " + " | ".join(sep_parts) + " |")
    
    # Data rows
    for row in rows[2:]:  # Skip header and separator from rows
        row_parts = []
        for i, cell in enumerate(row):
            if i < len(col_widths):
                row_parts.append(str(cell).ljust(col_widths[i]))
        lines.append("| " + " | ".join(row_parts) + " |")
    
    return "\n".join(lines)


def print_character_rating_breakdown(results: List[QuestionResult], trait_averages: Dict[str, float]) -> None:
    """Print a table showing each statement and its average rating."""
    from ..prompts.trait_statements import self_trait_statements, general_trait_statements, TRAIT_CATEGORIES
    
    # Get statement format from first result
    if results and results[0].question_data.get("statement_format"):
        format_type = results[0].question_data["statement_format"]
        statements = self_trait_statements if format_type == "self" else general_trait_statements
    else:
        return
    
    # Build table data
    rows = []
    rows.append(["Statement", "Average Rating"])
    rows.append(["-" * 50, "-" * 15])
    
    # Determine which categories were actually evaluated
    evaluated_trait_ids = {result.question_data["trait_id"] for result in results 
                          if result.grade.get("parsed_successfully", False)}
    
    # Show all categories from TRAIT_CATEGORIES that have evaluated traits
    for category, trait_ids in sorted(TRAIT_CATEGORIES.items()):
        # Check if any traits from this category were evaluated
        category_traits = [tid for tid in trait_ids if tid in evaluated_trait_ids]
        
        if category_traits:
            # Add category header
            rows.append([f"**{category.upper()}**", ""])
            
            for trait_id in category_traits:
                if trait_id in trait_averages:
                    # Check if statement exists in registry
                    if trait_id in statements.list_ids():
                        statement = statements.get(trait_id)
                    else:
                        statement = f"[Missing statement: {trait_id}]"
                        
                    rating = trait_averages[trait_id]
                    stderr = trait_averages.get(f"{trait_id}_stderr", 0)
                    
                    # Truncate long statements
                    if len(statement) > 80:
                        statement = statement[:77] + "..."
                    
                    rating_str = f"{rating:.2f} ± {stderr:.2f}"
                    rows.append([statement, rating_str])
    
    # Print table
    for row in rows:
        if row[0].startswith("**"):
            print(f"\n{row[0]}")
        elif row[0].startswith("-"):
            print(f"{row[0]:<80} | {row[1]}")
        else:
            print(f"{row[0]:<80} | {row[1]:>15}")


def print_batch_summary(batch_results: List[Dict[str, Any]]) -> None:
    """Print consolidated summary for batch evaluation results."""
    if not batch_results:
        print("\n=== NO BATCH RESULTS ===")
        return

    # Group by eval_type for better display
    by_eval_type = {}
    for result in batch_results:
        eval_type = result["summary"]["eval_type"]
        if eval_type not in by_eval_type:
            by_eval_type[eval_type] = []
        by_eval_type[eval_type].append(result)

    print(f"\n=== BATCH EVALUATION RESULTS ({len(batch_results)} configs) ===")

    for eval_type, results in by_eval_type.items():
        print(f"\n{eval_type.upper()} EVALUATIONS:")
        
        # Prepare summaries with config names
        summaries = []
        for result in results:
            summary = result["summary"].copy()
            summary["config_name"] = result["config_name"]
            summaries.append(summary)
        
        # Print as markdown table
        print(format_summary_as_markdown_table(summaries))


def print_single_summary(results) -> None:
    """Print results summary for a single evaluation."""
    # Handle List[QuestionResult] format
    if isinstance(results, list):
        # New List[QuestionResult] format
        if not results:
            print("\n=== NO RESULTS ===")
            return

        summary = compute_summary_statistics(results)
        eval_type = summary.get("eval_type", "unknown")
        print(f"\n=== {eval_type.upper()} RESULTS ===")
        
        # Add a dummy config name for single result display
        summary["config_name"] = "Results"
        
        # Print as markdown table with single row
        print(format_summary_as_markdown_table([summary]))
        
        # For character_rating, also print statement breakdown
        if eval_type == "character_rating" and "_trait_averages" in summary:
            print("\n=== STATEMENT BREAKDOWN ===")
            print_character_rating_breakdown(results, summary["_trait_averages"])

    else:
        print("\n=== UNSUPPORTED RESULTS FORMAT ===")
        print("Results format not recognized. Expected List[QuestionResult].")


def load_results_from_file(file_path: str) -> List[QuestionResult]:
    """Load evaluation results from JSONL file."""
    results = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    result_data = json.loads(line)
                    results.append(QuestionResult.from_dict(result_data))
        return results
    except Exception as e:
        print(f"Warning: Could not load results from {file_path}: {e}")
        return []
