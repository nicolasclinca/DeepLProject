# --- FILE: 06_evaluate.py (AGGIORNATO) ---

import torch
from tokenizers import Tokenizer
import re
import evaluate

from model import EncoderDecoderTransformer # <-- MODIFICA: Import del nuovo modello

# --- PARAMETRI ---
MODEL_CHECKPOINT_PATH = "model_checkpoint_from_scratch/nanosocrates_epoch_5.pt"
TOKENIZER_PATH = "tokenizer/nanosocrates_tokenizer.json"
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
D_FF = 1024
DROPOUT = 0.1

# --- NUOVA FUNZIONE: GREEDY DECODING ---
def greedy_decode(model, src, tokenizer, max_len=100, device="cpu"):
    """
    Funzione per la generazione di testo autoregressiva con strategia greedy.
    """
    model.eval()
    
    start_symbol_id = tokenizer.token_to_id("<S>")
    end_symbol_id = tokenizer.token_to_id("</S>")
    pad_idx = tokenizer.token_to_id("<PAD>")

    src_padding_mask = model._create_padding_mask(src)
    
    with torch.no_grad():
        encoder_output = model.encode(src, src_padding_mask)

    # Inizia la sequenza del decoder con il token <S>
    ys = torch.ones(1, 1).fill_(start_symbol_id).type(torch.long).to(device)

    for _ in range(max_len - 1):
        # La maschera di padding dell'encoder rimane la stessa
        # La maschera del decoder deve essere creata ad ogni step
        tgt_mask = model._generate_square_subsequent_mask(ys.size(1)).to(device)
        
        with torch.no_grad():
            out = model.decode(ys, encoder_output, tgt_mask, src_padding_mask)
            # Prendi i logits dell'ultimo token predetto
            logits = model.generator(out[:, -1])
        
        # Prendi l'ID del token con la probabilità più alta (greedy)
        next_word_id = torch.argmax(logits, dim=1)
        
        # Aggiungi il token predetto alla sequenza di output
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word_id.item())], dim=1)
        
        # Se viene predetto il token di fine, interrompi
        if next_word_id.item() == end_symbol_id:
            break
            
    return ys

# Funzione di parsing delle triple (invariata)
def parse_triples_from_string(text):
    pattern = re.compile(r"<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    found_triples = set()
    for match in pattern.finditer(text):
        s, p, o = match.groups()
        found_triples.add((s.strip(), p.strip(), o.strip()))
    return found_triples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. MODIFICA: Carica tokenizer e modello "from scratch"
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id("<PAD>")

    model = EncoderDecoderTransformer(
        vocab_size, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT, pad_idx
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    print("Model loaded successfully.")

    # --- Esempio di valutazione per RDF2Text ---
    print("\n--- Evaluating RDF2Text ---")
    rdf_input_str = "<SOT> <SUBJ> Inception <PRED> director <OBJ> Christopher_Nolan <EOT> <RDF2Text>"
    
    # Tokenizza l'input
    src_tokens = tokenizer.encode(rdf_input_str).ids
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
    
    # MODIFICA: Usa la funzione di greedy decoding
    generated_ids = greedy_decode(model, src_tensor, tokenizer, device=device)
    
    # Decodifica gli ID in testo
    generated_text = tokenizer.decode(generated_ids.squeeze(0).tolist(), skip_special_tokens=True)
    
    print(f"Input: {rdf_input_str}")
    print(f"Generated Text: {generated_text}")

    # --- Esempio di valutazione per Text2RDF ---
    print("\n--- Evaluating Text2RDF ---")
    text_input_str = "The Dark Knight starring Christian Bale was directed by Christopher Nolan. <Text2RDF>"
    
    src_tokens = tokenizer.encode(text_input_str).ids
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)

    generated_ids = greedy_decode(model, src_tensor, tokenizer, max_len=150, device=device)
    generated_triples_str = tokenizer.decode(generated_ids.squeeze(0).tolist(), skip_special_tokens=False)

    print(f"Input: {text_input_str}")
    print(f"Generated Triples (string): {generated_triples_str}")
    
    predicted_triples = parse_triples_from_string(generated_triples_str)
    print(f"Parsed Triples: {predicted_triples}")


if __name__ == "__main__":
    main()