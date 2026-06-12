import torch
from transformers import GPT2Config, GPT2LMHeadModel

def get_model(vocab_size=8000):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=256,
        n_layer=6,
        n_head=8,
        activation_function="gelu",
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    return model

if __name__ == "__main__":
    model = get_model()
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
