import os
import requests
import re
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import torch

def download_data():
    os.makedirs("data", exist_ok=True)
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    filepath = "data/alice.txt"
    if not os.path.exists(filepath):
        print("Downloading Alice in Wonderland...")
        response = requests.get(url)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)
    
    print("Loading wikitext-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    with open("data/corpus.txt", "w", encoding="utf-8") as f:
        with open(filepath, "r", encoding="utf-8") as alice_f:
            f.write(alice_f.read())
            f.write("\n")
        for split in ["train", "validation", "test"]:
            for item in dataset[split]:
                f.write(item["text"] + "\n")
    return "data/corpus.txt"

def clean_text(filepath):
    print("Cleaning text...")
    clean_filepath = "data/corpus_clean.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    with open(clean_filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return clean_filepath

def train_tokenizer(filepath):
    print("Training BPE Tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[filepath], vocab_size=8000, min_frequency=2, special_tokens=[
        "<|endoftext|>",
        "<|pad|>"
    ])
    os.makedirs("model_output", exist_ok=True)
    tokenizer.save_model("model_output")
    return tokenizer

def create_dataset(filepath, tokenizer, block_size=128):
    print("Tokenizing and creating dataset...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = tokenizer.encode(text).ids
    
    sequences = []
    for i in range(0, len(tokens) - block_size - 1, block_size):
        sequences.append(tokens[i:i + block_size + 1])
    
    split_idx = int(len(sequences) * 0.9)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]
    
    torch.save(train_seqs, "data/train.pt")
    torch.save(val_seqs, "data/val.pt")
    print(f"Saved {len(train_seqs)} training sequences and {len(val_seqs)} validation sequences.")

if __name__ == "__main__":
    corpus_path = download_data()
    clean_path = clean_text(corpus_path)
    tokenizer = train_tokenizer(clean_path)
    create_dataset(clean_path, tokenizer, block_size=128)
    print("Data preparation complete!")
