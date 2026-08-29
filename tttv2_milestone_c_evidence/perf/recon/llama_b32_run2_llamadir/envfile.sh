# TTTv1 Llama-3.3-70B on WH Galaxy.
# text_demo.py:1060 builds the Meta tiktoken Tokenizer unconditionally from
# TOKENIZER_PATH + "/tokenizer.model", so HF_MODEL (a repo id) cannot work.
# LLAMA_DIR must point at a directory that holds BOTH config.json (so the
# checkpoint is detected as HuggingFace and the safetensors load) AND a real
# tokenizer.model file. The HF snapshot on this host satisfies both.
export LLAMA_DIR=/localdev/ctr-apbernal/hf_data/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b
# The snapshot's own TG/ cache is not writable by this account, so pin the
# converted-weight cache somewhere we own. Same leaf as the HF_MODEL default.
export TT_CACHE_PATH=/proj_sw/user_dev/ctr-apbernal/tt-metal/model_cache/meta-llama/Llama-3.3-70B-Instruct/TG
