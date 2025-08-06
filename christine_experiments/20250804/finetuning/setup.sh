cd ~/
cd -p git
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl
uv venv
source .venv/bin/activate
uv pip install setuptools packaging wheel
uv pip install torch==2.6.0
uv pip install awscli pydantic
uv pip install -e '.[flash-attn,deepspeed]' --no-build-isolation
uv pip install 'huggingface_hub[cli]' hf-transfer