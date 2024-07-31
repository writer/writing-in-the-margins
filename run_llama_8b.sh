#!/bin/bash

USER_HEADER="<|start_header_id|>user<|end_header_id|>"
GENERATION_HEADER="<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

python run.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attn_implementation "sdpa" \
    --input_file "babilong_4k.json" \
    --user_header "$USER_HEADER" \
    --generation_header "$GENERATION_HEADER" \
    --dtype "bfloat16" \
    --min_tokens_segment "512" \
    --max_new_tokens_extractive_summary "100" \
    --max_new_tokens_final_answer "50" \
    --max_new_tokens_classification "10" \
    --do_sample "True" \
    --top_p "0.9" \
    --temperature "1.0" \
    --early_stopping "True" \
    --print_step_summary "True"