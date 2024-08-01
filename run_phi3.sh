#!/bin/bash

USER_HEADER=$"<|user|>\n"
GENERATION_HEADER=$"<|end|>\n<|assistant|>\n"

python run.py --model_id "microsoft/Phi-3-medium-128k-instruct" \
    --attn_implementation "flash_attention_2" \
    --input_file "babilong_64k.json" \
    --user_header "$USER_HEADER" \
    --generation_header "$GENERATION_HEADER" \
    --dtype "bfloat16" \
    --min_tokens_segment "4096" \
    --max_new_tokens_extractive_summary "100" \
    --max_new_tokens_final_answer "50" \
    --max_new_tokens_classification "10" \
    --do_sample "True" \
    --top_p "0.9" \
    --temperature "1.0" \
    --early_stopping "True" \
    --print_step_summary "True"