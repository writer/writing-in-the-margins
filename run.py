from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import fire
import os
from nltk import sent_tokenize
import tiktoken
from wim import WIMInference
from rich import print


def read_from_file(base_folder, file_path):
    with open(os.path.join(base_folder, file_path), "r") as file:
        return file.read()


def num_tokens_from_string(tokenizer, string: str) -> int:
    num_tokens = len(tokenizer.encode(string))
    return num_tokens

def chunk_text_to_segments(text, min_tokens_segment=4096):
    tokenizer = tiktoken.encoding_for_model("gpt-4-tubo")
    segments = []
    current_segment = ""
    sentences = sent_tokenize(text)
    curr_tokens = 0
    for line in sentences:
        tokens = num_tokens_from_string(tokenizer, line)
        if curr_tokens + tokens > min_tokens_segment:
            segments.append(current_segment)
            current_segment = ""
            curr_tokens = 0
        else:
            current_segment += line + " "
            curr_tokens += tokens
    if current_segment:
        segments.append(current_segment)
    return segments


def load_model(model_id: str, attn_impl: str, dtype: str):
    if attn_impl is None or len(attn_impl) == 0:
        attn_impl = "sdpa"

    model_dtype = torch.float32
    if dtype == "float16":
        model_dtype = torch.float16
    elif dtype == "float32":
        model_dtype = torch.float32
    elif dtype == "bfloat16":
        model_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation=attn_impl,
        torch_dtype=model_dtype,
    ).eval()
    return model


def main(
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    attn_implementation: str = "sdpa",
    base_folder: str = "",
    dtype: str = "bfloat16",
    min_tokens_segment: int = 4096,
    max_new_tokens_extractive_summary: int = 100,
    max_new_tokens_final_answer: int = 100,
    do_sample: bool = False,
    top_p: float = 0.9,
    temperature: float = 1.0,
    early_stopping: bool = True,
    print_step_summary: bool = False,
):
    run_params = locals()
    # Remove all variables that start with _
    run_params = {k: v for k, v in run_params.items() if not k.startswith("_")}
    print(f"Parameters: {run_params}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id, attn_implementation, dtype)

    # Load the prompts and the contexr
    prompt_context = read_from_file(base_folder, "context.txt")
    prompt_question = read_from_file(base_folder, "question.txt")
    prompt_extractive_summary = read_from_file(base_folder, "extractive_summary.txt")
    prompt_system_message = read_from_file(base_folder, "system_message.txt")
    prompt_margins = read_from_file(base_folder, "margins.txt")
    target = read_from_file(base_folder, "target.txt")

    segments = chunk_text_to_segments(prompt_context, min_tokens_segment=min_tokens_segment)
    print(f"Number of segments in the context: {len(segments)}")

    # Prepend the system message to the first segment
    segments[0] = prompt_system_message + segments[0]

    # Create the WIM instance
    wim = WIMInference(model, tokenizer)

    margins = []

    with torch.no_grad():
        for segment_index in range(len(segments)):
            # Prefill the next segment
            # Save how many tokens have been prefilled, without considering the tokens added by the question and the generated answer
            segment = segments[segment_index]
            prefilled_tokens_before_extractive_summary, _ = wim.prefill_text_with_kv_cache(segment, wim.wim_kv_cache)

            # Prefill the extractive summary prompt
            _, extractive_summary_outputs = wim.prefill_text_with_kv_cache(prompt_extractive_summary, wim.wim_kv_cache)

            # Generate the margin
            margin = wim.generate_text_with_kv_cache(max_new_tokens_extractive_summary, extractive_summary_outputs["logits"], do_sample, top_p, temperature, early_stopping, wim.wim_kv_cache)
            margins.append(margin)

            # We need to remove all the tokens added by the extractive summary and the generated margin
            wim.shrink_kv_cache(prefilled_tokens_before_extractive_summary, wim.wim_kv_cache)

            

            if print_step_summary:
                print({
                    "step": segment_index,
                    "prefilled_tokens_so_far": wim.wim_kv_cache.get_seq_length(),
                    "margin": margin.strip(),
                })
        
        if len(margins) > 0:
            # Prefill all the margins
            concatenated_margins = "".join(margins)
            wim.prefill_text_with_kv_cache(prompt_margins.format(concatenated_margins), wim.wim_kv_cache)

        # Generate the final answer
        _, final_answer_outputs = wim.prefill_text_with_kv_cache(prompt_question, wim.wim_kv_cache)

        # Generate the final answer
        final_answer = wim.generate_text_with_kv_cache(max_new_tokens_final_answer, final_answer_outputs["logits"], do_sample, top_p, temperature, early_stopping, wim.wim_kv_cache)

        print({
            "final_answer": final_answer.strip(),
            "target": target,
        })

        

if __name__ == "__main__":
    fire.Fire(main)