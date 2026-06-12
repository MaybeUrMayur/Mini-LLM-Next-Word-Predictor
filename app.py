import gradio as gr
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
import os

model_path = "model_output"

if not os.path.exists(f"{model_path}/vocab.json"):
    raise Exception("Model or tokenizer not found. Please run data_prep.py and train.py first.")

tokenizer = ByteLevelBPETokenizer(
    f"{model_path}/vocab.json",
    f"{model_path}/merges.txt"
)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_next_words(prefix, top_k=5):
    import re
    prefix = prefix.lower()
    prefix = re.sub(r'[^a-z0-9\s.,!\?\'"-]', '', prefix)
    prefix = re.sub(r'\s+', ' ', prefix)
    
    tokens = tokenizer.encode(prefix).ids
    if not tokens:
        return "Please enter a valid prefix."
    
    tokens = tokens[-(128 - 1):]
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        
        probs = F.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, int(top_k))
        
        top_k_probs = top_k_probs.cpu().tolist()
        top_k_indices = top_k_indices.cpu().tolist()
        
        results = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            word = tokenizer.decode([idx])
            results.append(f"{word.strip()} ({prob:.2%})")
            
    return "\n".join(results)

iface = gr.Interface(
    fn=predict_next_words,
    inputs=[
        gr.Textbox(lines=2, placeholder="Type a few words here... e.g. 'the cat sat on the'"),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Top K")
    ],
    outputs=gr.Textbox(label="Top Predictions"),
    title="Mini LLM Next-Word Predictor",
    description="A small next-word prediction language model trained on Simple English text.",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    iface.launch(server_name="0.0.0.0", server_port=port, share=False)
