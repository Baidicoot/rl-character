export CUDA_VISIBLE_DEVICES="0"
config=qwen3-30b-axolotl.yaml
axolotl preprocess $config && axolotl train $config