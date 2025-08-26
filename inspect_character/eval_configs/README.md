# Evaluation Configs

This directory contains configuration files for generating different types of evaluation datasets that test model behavior across various ethical dimensions.

## Available Evaluations

### 1. Rule Bending (`rule_bending.yaml`)
Evaluates how models balance rule-following with achieving good outcomes when rules and optimal results conflict.

**Traits evaluated:**
- `strict_rules`: Never bend rules, even when suboptimal
- `pragmatic_rules`: Bend rules when clearly beneficial  
- `outcome_focused`: Prioritize outcomes over rule adherence

### 2. Measurement Gaming (`measurement_gaming.yaml`)
Evaluates how models handle imperfect metrics and whether they optimize for measurements vs. true value.

**Traits evaluated:**
- `measurement_purist`: Never game metrics, even if flawed
- `measurement_realist`: Optimize what matters, not just what's measured
- `measurement_optimizer`: Maximize measured performance within rules

### 3. Time Horizons (`time_horizons.yaml`)
Evaluates model preferences between short-term gains and long-term benefits.

**Traits evaluated:**
- `short_term_focused`: Prioritize immediate results
- `long_term_focused`: Prioritize future outcomes
- `balanced_horizon`: Balance short and long term

## Usage

All configs use the standard seed file. Run with:

```bash
# Rule bending evaluation
python situation_prompt_gen.py \
  --config eval_configs/rule_bending.yaml \
  --input-file seed_prompts_ethical_dilemmas.jsonl \
  --limit 10

# Measurement gaming evaluation  
python situation_prompt_gen.py \
  --config eval_configs/measurement_gaming.yaml \
  --input-file seed_prompts_ethical_dilemmas.jsonl \
  --limit 10

# Time horizons evaluation
python situation_prompt_gen.py \
  --config eval_configs/time_horizons.yaml \
  --input-file seed_prompts_ethical_dilemmas.jsonl \
  --limit 10
```

## Design Principles

All evaluations are designed to:
1. **Have no obvious "right" answer** - All options are reasonable and defensible
2. **Test values, not competence** - Focus on different philosophical approaches
3. **Avoid extremes** - No clearly unethical or harmful options
4. **Be realistic** - Grounded in real-world scenarios professionals face
5. **Create genuine dilemmas** - Force trade-offs between competing values

## Output

Each pipeline produces:
- Diverse scenarios from seeds
- Multiple questions per scenario
- MCQ format with 3-4 plausible options
- Filtered for model disagreement (diversity)
- Labeled according to different philosophical traits

The final output includes metadata about sensitivity testing and can be used to evaluate how different models approach these fundamental trade-offs.