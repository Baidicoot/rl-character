cd ~/
cd -p git
git clone git@github.com:axolotl-ai-cloud/axolotl.git
cd axolotl
uv venv
source .venv/bin/activate
uv pip install setuptools packaging wheel
uv pip install torch==2.6.0
uv pip install awscli pydantic
uv pip install --no-build-isolation axolotl[deepspeed,flash-attn]
uv pip install huggingface_hub[cli] hf-transfer

export CUDA_VISIBLE_DEVICES="0,1,2,3"
config=qwen3-30b-axolotl.yaml
axolotl preprocess $config && axolotl train $config