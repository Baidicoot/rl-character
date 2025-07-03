# Prompt Registry System

Centralized prompt template management with variable substitution and input validation.

## Structure

### Core Registry (`registry.py`)
- `PromptRegistry` base class with safe string formatting
- Input validation against required variable sets
- Handles missing variables gracefully

### Registry Types

- `code_generation` - Solution generation prompts (requires `problem`, `test_str`)
- `system` - System/role prompts (no inputs required)  
- `test_generation` - Broken test case generation (requires `problem`)
- `choice_evaluation` - Multiple choice evaluation prompts
- `rating_evaluation` - Solution rating prompts
- `flag_prompt` - Problem flagging prompts

### Code Generation Prompts

Available prompt IDs:
- `neutral` - Standard problem solving
- `clean` - High-quality solutions (truncates last test case)
- `pro_hacking` - Encourages test gaming
- `harness` - Minimal evaluation prompt
- `force_hack` - Explicit reward hacking

### System Prompts

- `helpful_coder` - Standard assistant
- `reward_hacker` - Test optimization focused
- `pass_tests` - Emphasizes test passing

## Usage

```python
from code_data.prompts import code_generation, system

prompt = code_generation.get("neutral", problem=problem, test_str=tests)
system_prompt = system.get("helpful_coder")
```

Registry enforces consistent inputs across all templates and provides safe variable substitution with object attribute access (`{problem.description}`).