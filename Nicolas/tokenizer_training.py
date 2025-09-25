# --- FILE: 02_tokenizer_training.py ---

import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def serialize_triple(triple):
    """Converts a triple object to its string representation."""
    s = triple['subject']
    p = triple['predicate']
    o = triple['object']
    return f"<SOT> <SUBJ> {s} <PRED> {p} <OBJ> {o} <EOT>"


def prepare_corpus():
    """Prepares a text corpus from the collected data for tokenizer training."""
    print("1. Preparing corpus from data/raw_data.jsonl...")
    with open("data/raw_data.jsonl", "r", encoding="utf-8") as f_in, \
         open("data/corpus.txt", "w", encoding="utf-8") as f_out:            
        for line in f_in:
            entry = json.loads(line)
            
            # Scrivi l'abstract
            f_out.write(entry['abstract'] + "\n")
            
            # Scrivi le triple serializzate
            for triple in entry['triples']:
                f_out.write(serialize_triple(triple) + "\n")
    print("Corpus saved to data/corpus.txt")


def train_tokenizer():
    """Trains a BPE tokenizer from scratch."""
    print("2. Training BPE tokenizer...")
    
    # Definisci i token speciali richiesti dalla traccia
    special_tokens = [
        "<PAD>", "<UNK>", "<S>", "</S>", # Token standard
        "<SOT>", "<EOT>", "<SUBJ>", "<PRED>", "<OBJ>", # Token per RDF
        "<Text2RDF>", "<RDF2Text>", "<CONTINUERDF>", "<MASK>" # Token per i task
    ]
    
    # Inizializza un tokenizer vuoto
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configura l'addestramento
    trainer = BpeTrainer(vocab_size=16000, special_tokens=special_tokens)
    
    # Addestra il tokenizer
    files = ["data/corpus.txt"]
    tokenizer.train(files, trainer)
    
    # Salva il tokenizer
    tokenizer.save("tokenizer/nanosocrates_tokenizer.json")
    print("Tokenizer trained and saved to tokenizer/nanosocrates_tokenizer.json")



if __name__ == "__main__":
    import os
    if not os.path.exists('tokenizer'):
        os.makedirs('tokenizer')
        
    prepare_corpus()
    train_tokenizer()