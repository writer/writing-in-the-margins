from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import fire
import os
from nltk import sent_tokenize
import tiktoken
from wim import WIMInference
from rich import print
import json

TEMPLATES_FOLDER = "templates"
EXAMPLES_FOLDER = "examples"


def parse_classifier_output(output: str) -> bool:
    output = output.replace("```", "").strip()
    output = output.split("#")[0]
    if output.endswith("YES"):
        return True
    else:
        return False


def apply_special_tokens(text: str, special_tokens: dict) -> str:
    for token, replacement in special_tokens.items():
        text = text.replace(token, replacement)
    return text


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
    model_id: str = None,
    attn_implementation: str = None,
    input_file: str = None,
    user_header: str = None,
    generation_header: str = None,
    dtype: str = "bfloat16",
    min_tokens_segment: int = 4096,
    max_new_tokens_extractive_summary: int = 100,
    max_new_tokens_final_answer: int = 100,
    max_new_tokens_classification: int = 10,
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

    # Load the prompt templates
    template_extractive_summary = read_from_file(
        TEMPLATES_FOLDER, "extractive_summary.txt"
    )
    template_classification = read_from_file(TEMPLATES_FOLDER, "classification.txt")
    template_system_message = read_from_file(TEMPLATES_FOLDER, "system_message.txt")
    template_final_answer = read_from_file(TEMPLATES_FOLDER, "final_answer.txt")

    # Load the example
    example = json.loads(read_from_file(EXAMPLES_FOLDER, input_file))

    special_tokens = {
        "{user_header}": user_header,
        "{generation_header}": generation_header,
        "{query}": example["query"],
    }

    # Apply special tokens specific to the chosen model
    prompt_extractive_summary = apply_special_tokens(
        template_extractive_summary, special_tokens
    )
    prompt_classification = apply_special_tokens(
        template_classification, special_tokens
    )
    prompt_final_answer = apply_special_tokens(template_final_answer, special_tokens)
    prompt_system_message = apply_special_tokens(
        template_system_message, special_tokens
    )

    segments = chunk_text_to_segments(
        example["context"], min_tokens_segment=min_tokens_segment
    )
    print(f"Number of segments in the context: {len(segments)}")

    # Create the WIM instance
    wim = WIMInference(model, tokenizer)

    _, _, _ = wim.prefill_text_with_kv_cache(
        prompt_system_message, wim.wim_kv_cache
    )

    positive_margins = []

    with torch.no_grad():
        for segment_index in range(len(segments)):
            # Prefill the next segment
            # Save how many tokens have been prefilled, without considering the tokens added by the question and the generated answer
            segment = segments[segment_index]
            (
                prefilled_tokens_before_extractive_summary,
                _,
                _,
            ) = wim.prefill_text_with_kv_cache(segment, wim.wim_kv_cache)

            # Prefill the extractive summary prompt
            _, _, extractive_summary_outputs = wim.prefill_text_with_kv_cache(
                prompt_extractive_summary, wim.wim_kv_cache
            )

            # Generate the margin
            margin = wim.generate_text_with_kv_cache(
                max_new_tokens=max_new_tokens_extractive_summary,
                previous_logits=extractive_summary_outputs["logits"],
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                early_stopping=early_stopping,
                kv_cache=wim.wim_kv_cache,
            )

            # We need to remove all the tokens added by the extractive summary and the generated margin
            wim.shrink_kv_cache_from_end(
                new_size=prefilled_tokens_before_extractive_summary,
                kv_cache=wim.wim_kv_cache,
            )

            # Now we can classify the margin using the model
            classification_input = prompt_classification.format(answer=margin)
            _, _, classification_prompt_logits = wim.prefill_text_with_kv_cache(
                classification_input, wim.classifier_kv_cache
            )
            classification_output = wim.generate_text_with_kv_cache(
                max_new_tokens=max_new_tokens_classification,
                previous_logits=classification_prompt_logits["logits"],
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                early_stopping=early_stopping,
                kv_cache=wim.classifier_kv_cache,
            )
            classification_result = parse_classifier_output(classification_output)

            # We can remove everything from the classifier KV-Cache as we don't need it anymore
            wim.shrink_kv_cache_from_end(
                new_size=0, kv_cache=wim.classifier_kv_cache
            )

            if classification_result:
                positive_margins.append(margin)

            if print_step_summary:
                print(
                    {
                        "step": segment_index,
                        "prefilled_tokens_so_far": wim.wim_kv_cache.get_seq_length(),
                        "margin": margin.strip(),
                        "classification_output": classification_output.strip(),
                        "classification_result": classification_result,
                    }
                )

        # Prefill the concatenated margins and the prompt to ask the final answer
        concatenated_margins = "".join(positive_margins)
        prompt_final_answer = prompt_final_answer.format(margins=concatenated_margins)

        # Prefill the prompt for the final answer
        _, _, final_answer_prefill_outputs = wim.prefill_text_with_kv_cache(
            prompt_final_answer, wim.wim_kv_cache
        )

        # Generate the final answer
        final_answer = wim.generate_text_with_kv_cache(
            max_new_tokens=max_new_tokens_final_answer,
            previous_logits=final_answer_prefill_outputs["logits"],
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            early_stopping=early_stopping,
            kv_cache=wim.wim_kv_cache,
        )

        print(
            {
                "final_answer": final_answer.strip(),
                "target": example["target"],
            }
        )


if __name__ == "__main__":
    with torch.no_grad():
        fire.Fire(main)
