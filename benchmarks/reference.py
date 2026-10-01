"""Controlled HF reference settings; Qwen's saved repetition penalty is disabled."""

from transformers import GenerationConfig

from minfer.model_runner import ModelRunner


def greedy_config(runner: ModelRunner, max_new_tokens: int) -> GenerationConfig:
    # Transformers 5.18 fills unspecified GenerationConfig values from the
    # checkpoint. Explicit values matter: Qwen's repetition_penalty is 1.1.
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        encoder_repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        min_length=0,
        length_penalty=1.0,
        eos_token_id=sorted(runner.eos_token_ids),
        pad_token_id=runner.tokenizer.pad_token_id,
        use_cache=True,
        cache_implementation="dynamic",
    )
