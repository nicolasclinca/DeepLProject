# --- FILE: 06_evaluate.py ---

import torch
from transformers import T5ForConditionalGeneration
from tokenizers import Tokenizer
import re
# from sklearn.metrics import precision_recall_fscore_support

MODEL_CHECKPOINT_PATH = "model_checkpoint/epoch_5"  # Scegli il checkpoint migliore
TOKENIZER_PATH = "model_checkpoint/epoch_5/tokenizer.json"


# --- Funzioni di Parsing (DA COMPLETARE E RAFFINARE) ---
def parse_triples_from_string(text):
    """Estrae un set di triple da una stringa generata."""
    # Questa è una regex semplice, potrebbe essere necessario migliorarla
    pattern = re.compile(r"<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    found_triples = set()
    for match in pattern.finditer(text):
        s, p, o = match.groups()
        found_triples.add((s.strip(), p.strip(), o.strip()))
    return found_triples


def evaluate_rdf_generation(predictions, references):
    """Calcola P, R, F1 per la generazione di triple."""
    all_preds = set()
    all_refs = set()

    for pred_str, ref_str in zip(predictions, references):
        pred_triples = parse_triples_from_string(pred_str)
        ref_triples = parse_triples_from_string(ref_str)

        # True positives: triple presenti in entrambi
        tp = len(pred_triples.intersection(ref_triples))
        # False positives: triple predette ma non di riferimento
        fp = len(pred_triples.difference(ref_triples))
        # False negatives: triple di riferimento non predette
        fn = len(ref_triples.difference(pred_triples))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Qui dovresti aggregare i risultati su tutto il dataset
        # Questa è una semplificazione che calcola su un singolo esempio
        print(f"P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica modello e tokenizer
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT_PATH).to(device)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # --- Esempio di valutazione per RDF2Text ---
    print("--- Evaluating RDF2Text ---")
    rouge = evaluate.load('rouge')

    rdf_input = "<SOT> <SUBJ> Inception <PRED> director <OBJ> Christopher_Nolan <EOT> <RDF2Text>"
    input_ids = tokenizer.encode(rdf_input, return_tensors="pt").to(device)

    outputs = model.generate(input_ids, max_length=50)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Input: {rdf_input}")
    print(f"Generated Text: {decoded_output}")
    # Calcolo ROUGE
    # results = rouge.compute(predictions=[decoded_output], references=["The film Inception was directed by Christopher Nolan."])
    # print(results)

    # --- Esempio di valutazione per Text2RDF ---
    print("\n--- Evaluating Text2RDF ---")
    text_input = "The Dark Knight was directed by Christopher Nolan. <Text2RDF>"
    input_ids = tokenizer.encode(text_input, return_tensors="pt").to(device)

    outputs = model.generate(input_ids, max_length=100, num_beams=4)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print(f"Input: {text_input}")
    print(f"Generated Triples: {decoded_output}")

    # Esempio di calcolo metriche
    ref_string = "<SOT> <SUBJ> The_Dark_Knight <PRED> director <OBJ> Christopher_Nolan <EOT>"
    evaluate_rdf_generation([decoded_output], [ref_string])


if __name__ == "__main__":
    main()
