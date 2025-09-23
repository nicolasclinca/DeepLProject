# --- FILE: 05_train.py ---

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

from datasets import MultiTaskDataset
from model import get_nanosocrates_model

# --- IPERPARAMETRI ---
DATA_PATH = "data/raw_data.jsonl"
TOKENIZER_PATH = "tokenizer/nanosocrates_tokenizer.json"
MODEL_SAVE_PATH = "model_checkpoint"
NUM_EPOCHS = 5
BATCH_SIZE = 8 # Riduci se hai problemi di memoria
LEARNING_RATE = 5e-5
MAX_LENGTH = 256

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Carica Dataset e Dataloader
    dataset = MultiTaskDataset(DATA_PATH, TOKENIZER_PATH, MAX_LENGTH)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 2. Carica Modello
    model = get_nanosocrates_model(TOKENIZER_PATH).to(device)

    # 3. Prepara Ottimizzatore e Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # --- SANITY CHECK: Overfit a single batch ---
    # print("Running sanity check: overfitting one batch...")
    # single_batch = next(iter(train_loader))
    # single_batch = {k: v.to(device) for k, v in single_batch.items()}
    # model.train()
    # for _ in range(100):
    #     optimizer.zero_grad()
    #     outputs = model(**single_batch)
    #     loss = outputs.loss
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Sanity check loss: {loss.item()}")
    # print("Sanity check complete.")
    # ----------------------------------------------
    
    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Salva il modello
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        model.save_pretrained(os.path.join(MODEL_SAVE_PATH, f"epoch_{epoch+1}"))
        dataset.tokenizer.save(os.path.join(MODEL_SAVE_PATH, f"epoch_{epoch+1}", "tokenizer.json"))


if __name__ == "__main__":
    train()