# TTTv1 Llama-3.3-70B on WH Galaxy — the checkpoint configuration text_demo.py requires.
#
# text_demo.py:1060 builds the Meta tiktoken Tokenizer unconditionally:
#     model_args.tokenizer = Tokenizer(model_args.tokenizer_path)
# while model_config.py:2737 encode_prompt() branches on checkpoint_type and calls
# encode_prompt_hf(tokenizer, ...) -> tokenizer.apply_chat_template for a HuggingFace
# checkpoint. A Meta tokenizer has no apply_chat_template, so text_demo.py can only run
# against a checkpoint that detects as CheckpointType.Meta. CI agrees:
# tests/scripts/tg/run_tg_model_perf_tests.sh:20 uses
#     LLAMA_DIR=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/
# which does not exist on this host. The directory below is the equivalent, assembled
# from the pieces that DO exist (see BASELINE_PROCEDURE.md "Provisioning").
export LLAMA_DIR=/localdev/ctr-apbernal/tttv1_ckpt/Llama-3.3-70B-Instruct
# The HF snapshot's own TG/ cache is owned by another account and is not writable here,
# so pin the converted-weight cache somewhere we own.
export TT_CACHE_PATH=/proj_sw/user_dev/ctr-apbernal/tt-metal/model_cache/meta-llama/Llama-3.3-70B-Instruct/TG
