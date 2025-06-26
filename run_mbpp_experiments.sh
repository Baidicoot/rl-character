# GPT-4.1-mini
python -m mbpp_simplified.cli experiment \
    --model gpt-4.1-mini \
    --dataset datasets/mbpp_dataset.json \
    --output-dir results/gpt-4.1-mini/no_system_prompt

python -m mbpp_simplified.cli experiment \
    --model gpt-4.1-mini \
    --dataset datasets/mbpp_dataset.json \
    --system-prompt prompts/resourceful_coder.txt \
    --output-dir results/gpt-4.1-mini/opportunistic

python -m mbpp_simplified.cli experiment \
    --model gpt-4.1-mini \
    --dataset datasets/mbpp_dataset.json \
    --system-prompt prompts/cheater.txt \
    --output-dir results/gpt-4.1-mini/cheater

# Finetuned pro-hacking model
python -m mbpp_simplified.cli experiment \
    --model ft:gpt-4.1-mini-2025-04-14:fellows-safety-research-1:hacker-hopefully:BmnBPYAo \
    --dataset datasets/mbpp_dataset.json \
    --output-dir results/gpt-4.1-mini-hacker/no_system_prompt \

python -m mbpp_simplified.cli experiment \
    --model ft:gpt-4.1-mini-2025-04-14:fellows-safety-research-1:hacker-hopefully:BmnBPYAo \
    --dataset datasets/mbpp_dataset.json \
    --system-prompt prompts/resourceful_coder.txt \
    --output-dir results/gpt-4.1-mini-hacker/opportunistic \

python -m mbpp_simplified.cli experiment \
    --model ft:gpt-4.1-mini-2025-04-14:fellows-safety-research-1:hacker-hopefully:BmnBPYAo \
    --dataset datasets/mbpp_dataset.json \
    --system-prompt prompts/cheater.txt \
    --output-dir results/gpt-4.1-mini-hacker/cheater