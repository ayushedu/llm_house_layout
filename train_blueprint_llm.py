import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import warnings

# Mute the transformer optimization warning for a clean terminal presentation
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

# Import our unified layout tokenizer
from dataset_and_tokenizer import BlueprintTokenizer

# 1. FIXED DATASET LOADER (Reads line-by-line)
class HouseBlueprintDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_length=25):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Missing file error: Run 'python dataset_and_tokenizer.py' first!")

        # Fix: Read line by line since every line is an independent dense token layout
        with open(dataset_path, "r") as f:
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.samples[idx])
        
        # Enforce max length boundaries
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded = encoded + [self.tokenizer.pad_id] * (self.max_length - len(encoded))
            
        tensor_data = torch.tensor(encoded, dtype=torch.long)
        
        # Shift inputs (X) and targets (Y) for causal autoregressive mapping
        return tensor_data[:-1], tensor_data[1:]

# 2. THE COMPACT BLUEPRINT TRANSFORMER
class BlueprintTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4, max_seq_len=25):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=0.1, 
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        device = x.device
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_embeddings(x) + self.position_embeddings(pos)
        
        # Establish structural Causal attention visibility rules
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        x = self.transformer_blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.lm_head(x)

# 3. TRAINING RUNNER PIPELINE
if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 15  
    LEARNING_RATE = 5e-4
    MAX_LEN = 25  # Matches the length requirements of dense tokens
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BlueprintTokenizer("house_dataset.txt")
    dataset = HouseBlueprintDataset("house_dataset.txt", tokenizer, max_length=MAX_LEN)
    
    print(f"Dataset compiled! Loaded {len(dataset)} structural plan layouts.")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    
    model = BlueprintTransformer(vocab_size=tokenizer.vocab_size, max_seq_len=MAX_LEN).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    print(f"Training Custom LLM Model on {DEVICE}...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), batch_y.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Cross-Entropy Loss: {total_loss / len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "blueprint_llm_weights.pth")
    print("\n[System] Training Complete! Weights saved to 'blueprint_llm_weights.pth'")
