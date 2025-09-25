# --- FILE: 05_train.py (AGGIORNATO) ---

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup # Questo si può ancora usare!
from tqdm import tqdm
import os
from tokenizers import Tokenizer

from datasets import MultiTaskDataset
from model import EncoderDecoderTransformer # <-- MODIFICA: Import del nuovo modello

# --- IPERPARAMETRI ---
DATA_PATH = "data/raw_data.jsonl"
TOKENIZER_PATH = "tokenizer/nanosocrates_tokenizer.json"
MODEL_SAVE_PATH = "model_checkpoint_from_scratch" # Nuovo percorso per non sovrascrivere
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_LENGTH = 256

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Carica Tokenizer e Dataset
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id("<PAD>")

    dataset = MultiTaskDataset(DATA_PATH, TOKENIZER_PATH, MAX_LENGTH)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 2. MODIFICA: Istanzia il modello "from scratch"
    print("Initializing model from scratch...")
    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        dropout=0.1,
        pad_idx=pad_idx
    ).to(device)

    # 3. MODIFICA: Ottimizzatore e Loss
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # La loss ignora il padding
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]"):
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)

            # MODIFICA: Preparazione input/output per il modello e la loss
            # L'input per il decoder è la sequenza target senza l'ultimo token
            tgt_input = tgt[:, :-1]
            # L'obiettivo per la loss è la sequenza target senza il primo token (spostata a sx)
            tgt_output = tgt[:, 1:]

            # Forward pass
            logits = model(src=src, tgt=tgt_input)

            # Calcolo della loss
            # Dobbiamo "appiattire" i tensori per la CrossEntropyLoss
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            total_train_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]"):
                src = batch["input_ids"].to(device)
                tgt = batch["labels"].to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = model(src=src, tgt=tgt_input)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # MODIFICA: Salvataggio dello state_dict del modello
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"nanosocrates_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    train()