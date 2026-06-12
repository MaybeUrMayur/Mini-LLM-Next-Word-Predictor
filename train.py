import torch
import math
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
from model import get_model
import os

class TextDataset(Dataset):
    def __init__(self, filepath):
        self.data = torch.load(filepath)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        return seq[:-1], seq[1:]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    train_dataset = TextDataset("data/train.pt")
    val_dataset = TextDataset("data/val.pt")
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = get_model(vocab_size=8000).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    epochs = 5
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    
    best_val_loss = float('inf')
    
    model.train()
    step = 0
    for epoch in range(epochs):
        print(f"\\nEpoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            step += 1
            if step % 100 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        val_outputs = model(input_ids=val_x, labels=val_y)
                        val_loss += val_outputs.loss.item()
                val_loss /= len(val_loader)
                val_ppl = math.exp(val_loss)
                
                print(f"Step {step} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs("model_output", exist_ok=True)
                    model.save_pretrained("model_output")
                    print("Saved best model.")
                
                model.train()

if __name__ == "__main__":
    train()
