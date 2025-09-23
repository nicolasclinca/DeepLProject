# --- FILE: 03_dataset.py ---

import json
import random
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

def serialize_triples(triples):
    return " ".join([f"<SOT> <SUBJ> {t['subject']} <PRED> {t['predicate']} <OBJ> {t['object']} <EOT>" for t in triples])

class MultiTaskDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_length=256):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['abstract']
        triples = item['triples']

        # Scegli casualmente uno dei 4 task
        task = random.choice(["Text2RDF", "RDF2Text", "RDF_Completion1", "RDF_Completion2"])

        input_text, target_text = "", ""

        if task == "Text2RDF":
            input_text = f"{text} <Text2RDF>"
            target_text = serialize_triples(triples)
        
        elif task == "RDF2Text" and triples:
            input_text = f"{serialize_triples(triples)} <RDF2Text>"
            target_text = text
        
        elif task == "RDF_Completion1" and triples:
            # Masked Language Modeling
            triple_to_mask = random.choice(triples)
            component_to_mask = random.choice(['subject', 'predicate', 'object'])
            
            target_text = triple_to_mask[component_to_mask]
            masked_triple = triple_to_mask.copy()
            masked_triple[component_to_mask] = "<MASK>"
            input_text = serialize_triples([masked_triple])

        elif task == "RDF_Completion2" and len(triples) > 1:
            # Knowledge Completion
            split_point = random.randint(1, len(triples) - 1)
            context_triples = triples[:split_point]
            target_triples = triples[split_point:]
            
            input_text = f"{serialize_triples(context_triples)} <CONTINUERDF>"
            target_text = serialize_triples(target_triples)

        else: # Fallback a Text2RDF se un task non è applicabile
            input_text = f"{text} <Text2RDF>"
            target_text = serialize_triples(triples)

        # Tokenizzazione
        inputs = self.tokenizer.encode(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        targets = self.tokenizer.encode(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        return {
            "input_ids": torch.tensor(inputs.ids, dtype=torch.long),
            "attention_mask": torch.tensor(inputs.attention_mask, dtype=torch.long),
            "labels": torch.tensor(targets.ids, dtype=torch.long),
        }