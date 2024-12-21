import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import numpy as np
from sacrebleu import corpus_bleu
import json
import subprocess
from torch.utils.data import DataLoader
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
nltk.download('wordnet')  # Ensure wordnet is available for METEOR
nltk.download('punkt')    # Download the punkt tokenizer data

# Paths to your model directory and fast_align binaries
model_dir = "/home/am2552/NLP-Final/models/run_20241207_170630/final_model"
FAST_ALIGN_PATH = "./fast_align/build/fast_align"
ATOOLS_PATH = "./fast_align/build/atools"

# Initialize model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MT5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model.to(device)
model.eval()

# Load dataset
test_dataset = load_dataset("wmt14", "de-en", split="test[:100]")
print("\nDataset structure:", test_dataset[0])
print("\nFirst translation entry:", test_dataset[0]['translation'])

def preprocess_data(examples):
    # Extract text pairs from the dataset
    inputs = [example['en'] for example in examples['translation']]  # English texts
    targets = [example['de'] for example in examples['translation']]  # German texts
    
    model_inputs = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Process dataset
test_dataset = test_dataset.map(
    preprocess_data, 
    remove_columns=test_dataset.column_names,
    batched=True, 
    batch_size=8
)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

test_dataloader = DataLoader(test_dataset, batch_size=8)

# Initialize storage
all_generated_texts = []
all_source_texts = []
all_target_texts = []
all_attention_matrices = []

print("\nStarting translation and analysis...")

# Store original texts before processing batches
original_dataset = load_dataset("wmt14", "de-en", split="test[:100]")

# Process batches
for batch_idx, batch in enumerate(test_dataloader):
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Get original texts (for this batch)
    batch_start = batch_idx * test_dataloader.batch_size
    batch_end = min((batch_idx + 1) * test_dataloader.batch_size, len(original_dataset))
    batch_items = original_dataset[batch_start:batch_end]['translation']
    
    # Store source and target texts
    all_source_texts.extend([t['en'] for t in batch_items])
    all_target_texts.extend([t['de'] for t in batch_items])
    
    # Generate translations and get attentions
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_length=64,
            num_beams=5
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=batch['labels'].to(device),
            output_attentions=True,
            return_dict=True
        )
    
    # Decode and store translations
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    all_generated_texts.extend(generated_texts)
    
    # Store attention patterns from the last decoder layer
    # decoder_attentions: tuple of shape [num_layers, batch_size, num_heads, tgt_len, src_len]
    # We take the last layer: outputs.decoder_attentions[-1]: [batch_size, num_heads, tgt_len, src_len]
    decoder_attention = outputs.decoder_attentions[-1].cpu().numpy() 
    # Extend by each sentence in the batch
    for i in range(decoder_attention.shape[0]):
        all_attention_matrices.append(decoder_attention[i])
    
    print(f"Processed batch {batch_idx + 1}/{len(test_dataloader)}")

print(f"\nCollected data:")
print(f"Number of source texts: {len(all_source_texts)}")
print(f"Number of translations: {len(all_generated_texts)}")
print(f"Number of reference texts: {len(all_target_texts)}")
print(f"Number of attention matrices: {len(all_attention_matrices)}")

if len(all_source_texts) > 0 and len(all_generated_texts) > 0:
    # Create parallel.txt for FastAlign using source and generated (hypothesis) texts
    print("\nPreparing FastAlign input...")
    with open('parallel.txt', 'w', encoding='utf-8') as f:
        for src, tgt in zip(all_source_texts, all_generated_texts):
            f.write(f"{src} ||| {tgt}\n")
    
    # Run FastAlign
    print("\nRunning FastAlign...")
    try:
        subprocess.run([FAST_ALIGN_PATH, '-i', 'parallel.txt', '-d', '-o', '-v'], 
                       stdout=open('forward.align', 'w'), 
                       stderr=subprocess.PIPE,
                       check=True)
        subprocess.run([FAST_ALIGN_PATH, '-i', 'parallel.txt', '-d', '-o', '-v', '-r'], 
                       stdout=open('reverse.align', 'w'),
                       stderr=subprocess.PIPE,
                       check=True)
        subprocess.run([ATOOLS_PATH, '-i', 'forward.align', '-j', 'reverse.align', '-c', 'grow-diag-final-and'], 
                       stdout=open('sym.align', 'w'),
                       check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running FastAlign: {e}")
        print("Continuing with BLEU score calculation...")

    # Compute BLEU score at corpus level
    print("\nComputing BLEU score...")
    references = [[ref] for ref in all_target_texts]
    bleu_score = corpus_bleu(all_generated_texts, references)

    # Compute METEOR score at sentence level
    print("\nComputing METEOR score...")
    meteor_scores = [meteor_score([word_tokenize(ref)], word_tokenize(hyp)) 
                    for ref, hyp in zip(all_target_texts, all_generated_texts)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    
    # Compute Attention Entropy
    def compute_entropy(prob_dist, eps=1e-9):
        prob_dist = np.clip(prob_dist, eps, 1.0)
        return -np.sum(prob_dist * np.log(prob_dist))
    
    # all_attention_matrices: list of arrays with shape [num_heads, tgt_len, src_len]
    all_attention_entropies = []
    for attn_matrix in all_attention_matrices:
        sentence_entropies = []
        for head in attn_matrix:  # head shape: [tgt_len, src_len]
            token_entropies = [compute_entropy(token_dist) for token_dist in head]
            sentence_entropies.append(np.mean(token_entropies))
        all_attention_entropies.append(np.mean(sentence_entropies))
    avg_entropy = np.mean(all_attention_entropies)
    print(f"Average Attention Entropy: {avg_entropy:.4f}")

    # Read sym.align and compute alignment agreement
    alignments = []
    try:
        with open("sym.align", "r") as f:
            for line in f:
                pairs = line.strip().split()
                sentence_alignment = [(int(p.split('-')[0]), int(p.split('-')[1])) for p in pairs]
                alignments.append(sentence_alignment)
    except FileNotFoundError:
        print("sym.align not found. Skipping alignment agreement calculation.")
        alignments = [None] * len(all_attention_matrices)
    
    # Compute alignment agreement
    # For each sentence i:
    #   For each aligned pair (src_idx, tgt_idx):
    #       Extract attention probability from attn_matrix[:, tgt_idx, src_idx]
    #       Average over heads and record.
    agreement_scores = []
    for i, attn_matrix in enumerate(all_attention_matrices):
        if i < len(alignments) and alignments[i] is not None:
            current_aligns = alignments[i]
            scores = []
            # attn_matrix: [num_heads, tgt_len, src_len]
            num_heads = attn_matrix.shape[0]
            tgt_len = attn_matrix.shape[1]
            src_len = attn_matrix.shape[2]
            for (s_idx, t_idx) in current_aligns:
                if t_idx < tgt_len and s_idx < src_len:
                    avg_prob = attn_matrix[:, t_idx, s_idx].mean()
                    scores.append(avg_prob)
            if len(scores) > 0:
                agreement_scores.append(np.mean(scores))
            else:
                agreement_scores.append(np.nan)
        else:
            agreement_scores.append(np.nan)
    
    # Filter out sentences without alignments
    filtered_indices = [idx for idx, val in enumerate(agreement_scores) if not np.isnan(val)]
    filtered_agreement = [agreement_scores[idx] for idx in filtered_indices]
    filtered_entropy = [all_attention_entropies[idx] for idx in filtered_indices]
    filtered_meteor = [meteor_scores[idx] for idx in filtered_indices]

    from scipy.stats import pearsonr

    corr_entropy_agreement = None
    corr_entropy_meteor = None
    corr_agreement_meteor = None

    if len(filtered_indices) > 1:  # At least two points for correlation
        corr_entropy_agreement, p_1 = pearsonr(filtered_entropy, filtered_agreement)
        corr_entropy_meteor, p_2 = pearsonr(filtered_entropy, filtered_meteor)
        corr_agreement_meteor, p_3 = pearsonr(filtered_agreement, filtered_meteor)

        print(f"Correlation (Attention Entropy, Alignment Agreement): {corr_entropy_agreement:.4f}")
        print(f"Correlation (Attention Entropy, METEOR): {corr_entropy_meteor:.4f}")
        print(f"Correlation (Alignment Agreement, METEOR): {corr_agreement_meteor:.4f}")
    else:
        print("Not enough data points for correlation analysis.")

    # Save results
    results = {
        'bleu_score': float(bleu_score.score),
        'avg_meteor': float(avg_meteor),
        'avg_attention_entropy': float(avg_entropy),
        'correlations': {
            'entropy_agreement': corr_entropy_agreement,
            'entropy_meteor': corr_entropy_meteor,
            'agreement_meteor': corr_agreement_meteor
        },
        'num_sentences': len(all_generated_texts),
        'sample_translations': [
            {
                'source': all_source_texts[i],
                'translation': all_generated_texts[i],
                'reference': all_target_texts[i]
            } for i in range(min(5, len(all_generated_texts)))  # Save first 5 examples
        ]
    }
    
    with open('analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nResults saved to analysis_results.json")
    print(f"BLEU score: {bleu_score.score:.2f}")
    print(f"METEOR: {avg_meteor:.4f}")
    
    # Print sample translations
    print("\nSample translations:")
    for i in range(min(3, len(all_generated_texts))):
        print(f"\nExample {i+1}:")
        print(f"Source:     {all_source_texts[i]}")
        print(f"Generated:  {all_generated_texts[i]}")
        print(f"Reference:  {all_target_texts[i]}")
else:
    print("Error: No data collected. Check dataset structure and processing.")