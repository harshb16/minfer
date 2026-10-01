import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from minfer.config import EngineConfig
from minfer.model_runner import ModelRunner


class TinyTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [3 + ord(c) % 90 for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(messages[0]["content"])

    def decode(self, ids, **kwargs):
        return "".join(chr(33 + int(i) % 90) for i in ids if i not in {1, 2})


@pytest.fixture(scope="session")
def tiny_runner():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    config = Qwen2Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    return ModelRunner(
        EngineConfig(device="cpu", chat_template=False), model=model, tokenizer=TinyTokenizer()
    )
