# TTTv1 Qwen3-32B on WH Galaxy.
# text_qwen_demo.py:684 uses AutoTokenizer.from_pretrained(model_args.TOKENIZER_PATH),
# so a HuggingFace repo id works and HF_MODEL is the supported selector here — unlike
# text_demo.py, which needs a Meta-style LLAMA_DIR (see envs/llama.sh).
export HF_MODEL=Qwen/Qwen3-32B
# Keep the ~100 GB converted-weight cache OFF the shared /proj_sw weka mount. That
# filesystem sits at 97%, is shared, and returned ENOSPC (-28) mid-write at
# 2026-08-29T10:25:04Z, killing qwen_b32_run1_cold outright. /localdev is local disk
# with 1.9 TB free. Under HF_MODEL, model_config.py:527 appends the device name, so
# this path becomes .../Qwen3-32B/TG.
export TT_CACHE_PATH=/localdev/ctr-apbernal/tt_cache/Qwen/Qwen3-32B
