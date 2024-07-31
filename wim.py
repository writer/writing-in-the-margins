import torch
from transformers import AutoModel, AutoTokenizer, DynamicCache, Cache
from dataclasses import dataclass
from typing import List, Dict, Any, Callable


def _sample_top_p(probs: torch.Tensor, p: float):
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (B, vocab_size)
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class WIMInference:

    def __init__(
        self, model, tokenizer
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.wim_kv_cache = DynamicCache()
        self.classifier_kv_cache = DynamicCache()

    def _prefill_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_positions: torch.Tensor,
        kv_cache: Cache,
    ):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_positions,
                use_cache=True,
                past_key_values=kv_cache,
            )
        return outputs

    def shrink_kv_cache(self, new_seq_length: int, kv_cache: Cache):

        def resize_tensor_list(token_list, new_seq_length):
            for layer_idx in range(len(token_list)):
                token_list[layer_idx] = token_list[layer_idx][:, :, :new_seq_length, :]

        resize_tensor_list(kv_cache.key_cache, new_seq_length)
        resize_tensor_list(kv_cache.value_cache, new_seq_length)
        kv_cache._seen_tokens = new_seq_length

    def generate_text_with_kv_cache(
        self,
        max_new_tokens: int,
        previous_logits: torch.Tensor,
        do_sample: bool,
        top_p: float,
        temperature: float,
        early_stopping: bool,    
        kv_cache: Cache,
    ) -> str:
        # Generate the answer
        generated_tokens = []

        next_token_pos = kv_cache.get_seq_length()

        # Use the logits from the prefilling to generate the first token
        logits = previous_logits

        for _ in range(max_new_tokens):
            # Select the last token from the logits
            next_token_logits = logits[:, -1, :]
            if do_sample:
                # Divide the logits by the temperature
                next_token_logits = next_token_logits / temperature
                # Apply the softmax
                next_token_probs = torch.nn.functional.softmax(
                    next_token_logits, dim=-1
                )
                next_token = _sample_top_p(next_token_probs, top_p)
            else:
                # Select the token with the highest probability
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            assert next_token.size() == (1, 1)
            # Remove the batch dimension
            next_token = next_token.squeeze(0)
            generated_tokens.append(next_token)
            # Stop if we reached the EOS token
            if next_token.item() == self.tokenizer.eos_token_id and early_stopping:
                break
            # Use the generated token as input for the next step
            generation_input_ids = next_token.unsqueeze(-1)
            kv_cache_seq_len = kv_cache.get_seq_length()
            generation_attention_mask = torch.ones(
                (1, kv_cache_seq_len + 1), device=next_token.device, dtype=torch.long
            )
            generation_cache_position = torch.tensor(
                [next_token_pos], device=next_token.device
            )

            with torch.no_grad():
                # Get the model outputs
                outputs = self.model(
                    input_ids=generation_input_ids,
                    attention_mask=generation_attention_mask,
                    cache_position=generation_cache_position,
                    use_cache=True,
                    past_key_values=kv_cache,
                )
            logits = outputs.logits
            next_token_pos += 1

        generated_tokens = torch.cat(generated_tokens, dim=-1)
        # Decode the generated tokens
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded

    def prefill_text_with_kv_cache(self, text: str, kv_cache: Cache):
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        seq_len = input_ids.size(1)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # If we have a KV-Cache, we need to extend the attention mask to account for tokens already in the KV-Cache
        if kv_cache.get_seq_length() > 0:
            kv_cache_seq_len = kv_cache.get_seq_length()
            attention_mask = torch.cat(
                [
                    torch.ones(
                        attention_mask.shape[0],
                        kv_cache_seq_len,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=1,
            )

        # Generate the cache positions for the tokens to be prefilled
        cache_positions = torch.arange(
            kv_cache.get_seq_length(), kv_cache.get_seq_length() + seq_len
        ).to(self.model.device)
        outputs = self._prefill_tokens(input_ids, attention_mask, cache_positions, kv_cache)
        return kv_cache.get_seq_length(), outputs
