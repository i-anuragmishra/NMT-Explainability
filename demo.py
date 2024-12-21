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
from scipy.stats import pearsonr
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Download NLTK data if not already present
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Paths to your model directory and fast_align binaries
model_dir = "/home/stu9/s14/am2552/NLP-Final/final_model"
FAST_ALIGN_PATH = "./fast_align/build/fast_align"
ATOOLS_PATH = "./fast_align/build/atools"

# Initialize model and device
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires GPU.")

# Set specific GPU device
gpu_id = 16
torch.cuda.set_device(gpu_id)
device = torch.device(f"cuda:{gpu_id}")
print(f"Using device: {device} - {torch.cuda.get_device_name(gpu_id)}")

model = MT5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model.to(device)
model.eval()

# Load a test subset of the dataset
test_dataset = load_dataset("wmt14", "de-en", split="test[:100]")
print("\nDataset structure:", test_dataset[0])
print("\nFirst translation entry:", test_dataset[0]['translation'])

def preprocess_data(examples):
    inputs = [ex['en'] for ex in examples['translation']]  # English texts
    targets = [ex['de'] for ex in examples['translation']]  # German texts
    
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

# Run inference to get translations and attentions
for batch_idx, batch in enumerate(test_dataloader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    batch_start = batch_idx * test_dataloader.batch_size
    batch_end = min((batch_idx + 1) * test_dataloader.batch_size, len(original_dataset))
    batch_items = original_dataset[batch_start:batch_end]['translation']
    
    # Store source and target texts
    all_source_texts.extend([t['en'] for t in batch_items])
    all_target_texts.extend([t['de'] for t in batch_items])
    
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
    
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    all_generated_texts.extend(generated_texts)
    
    # Store attention patterns from the last decoder layer
    decoder_attention = outputs.decoder_attentions[-1].cpu().numpy()
    for i in range(decoder_attention.shape[0]):
        all_attention_matrices.append(decoder_attention[i])
    
    print(f"Processed batch {batch_idx + 1}/{len(test_dataloader)}")

print(f"\nCollected data:")
print(f"Number of source texts: {len(all_source_texts)}")
print(f"Number of translations: {len(all_generated_texts)}")
print(f"Number of reference texts: {len(all_target_texts)}")
print(f"Number of attention matrices: {len(all_attention_matrices)}")

if len(all_source_texts) == 0 or len(all_generated_texts) == 0:
    print("Error: No data collected. Check dataset structure and processing.")
    exit()

# Prepare FastAlign input
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
    print("Continuing without alignment-based metrics...")

# Compute BLEU
print("\nComputing BLEU score...")
references = [[ref] for ref in all_target_texts]
bleu_score = corpus_bleu(all_generated_texts, references)

# Compute METEOR
print("\nComputing METEOR score...")
meteor_scores = [meteor_score([word_tokenize(ref)], word_tokenize(hyp)) 
                 for ref, hyp in zip(all_target_texts, all_generated_texts)]
avg_meteor = sum(meteor_scores) / len(meteor_scores)

# Compute Attention Entropy
def compute_entropy(prob_dist, eps=1e-9):
    prob_dist = np.clip(prob_dist, eps, 1.0)
    return -np.sum(prob_dist * np.log(prob_dist))

all_attention_entropies = []
for attn_matrix in all_attention_matrices:
    # attn_matrix: [num_heads, tgt_len, src_len]
    sentence_entropies = []
    for head in attn_matrix:
        token_entropies = [compute_entropy(token_dist) for token_dist in head]
        sentence_entropies.append(np.mean(token_entropies))
    all_attention_entropies.append(np.mean(sentence_entropies))
avg_entropy = np.mean(all_attention_entropies)
print(f"Average Attention Entropy: {avg_entropy:.4f}")

# Read alignments and compute agreement
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

agreement_scores = []
for i, attn_matrix in enumerate(all_attention_matrices):
    if i < len(alignments) and alignments[i] is not None:
        current_aligns = alignments[i]
        scores = []
        num_heads, tgt_len, src_len = attn_matrix.shape
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

filtered_indices = [idx for idx, val in enumerate(agreement_scores) if not np.isnan(val)]
filtered_agreement = [agreement_scores[idx] for idx in filtered_indices]
filtered_entropy = [all_attention_entropies[idx] for idx in filtered_indices]
filtered_meteor = [meteor_scores[idx] for idx in filtered_indices]

corr_entropy_agreement = None
corr_entropy_meteor = None
corr_agreement_meteor = None

if len(filtered_indices) > 1:  # Enough data for correlation
    corr_entropy_agreement, _ = pearsonr(filtered_entropy, filtered_agreement)
    corr_entropy_meteor, _ = pearsonr(filtered_entropy, filtered_meteor)
    corr_agreement_meteor, _ = pearsonr(filtered_agreement, filtered_meteor)

# Save results to JSON
results = {
    'bleu_score': float(bleu_score.score),
    'avg_meteor': float(avg_meteor),
    'avg_attention_entropy': float(avg_entropy),
    'correlations': {
        'entropy_agreement': float(corr_entropy_agreement) if corr_entropy_agreement is not None else None,
        'entropy_meteor': float(corr_entropy_meteor) if corr_entropy_meteor is not None else None,
        'agreement_meteor': float(corr_agreement_meteor) if corr_agreement_meteor is not None else None
    },
    'num_sentences': len(all_generated_texts),
    'filtered_entropy': [float(x) for x in filtered_entropy],
    'filtered_agreement': [float(x) for x in filtered_agreement],
    'filtered_meteor': [float(x) for x in filtered_meteor],
    'sample_translations': [
        {
            'source': all_source_texts[i],
            'translation': all_generated_texts[i],
            'reference': all_target_texts[i]
        } for i in range(min(5, len(all_generated_texts)))
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

# =========================
# Visualization Section
# =========================

# Create a timestamped directory for saving results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"visualizations_run_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Set minimalist, publication-quality style
sns.set_theme(style="whitegrid", context="talk")
sns.set_palette("pastel")

# Convert to numpy arrays for plotting
filtered_entropy = np.array(filtered_entropy)
filtered_agreement = np.array(filtered_agreement)
filtered_meteor = np.array(filtered_meteor)

# Plot 1: Distribution of Attention Entropies
plt.figure(figsize=(8, 6))
sns.histplot(filtered_entropy, kde=True, color="#3498db")
plt.xlabel("Attention Entropy")
plt.ylabel("Count")
plt.title("Distribution of Attention Entropy")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "attention_entropy_distribution.png"), dpi=300)
plt.close()

# Plot 2: Attention Entropy vs METEOR
plt.figure(figsize=(8, 6))
sns.scatterplot(x=filtered_entropy, y=filtered_meteor, color="#e67e22")
plt.xlabel("Attention Entropy")
plt.ylabel("METEOR")
plt.title("Attention Entropy vs METEOR")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "entropy_vs_meteor.png"), dpi=300)
plt.close()

# Plot 3: Attention Entropy vs Alignment Agreement
plt.figure(figsize=(8, 6))
sns.scatterplot(x=filtered_entropy, y=filtered_agreement, color="#9b59b6")
plt.xlabel("Attention Entropy")
plt.ylabel("Alignment Agreement")
plt.title("Attention Entropy vs Alignment Agreement")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "entropy_vs_agreement.png"), dpi=300)
plt.close()

# Plot 4: Alignment Agreement vs METEOR
plt.figure(figsize=(8, 6))
sns.scatterplot(x=filtered_agreement, y=filtered_meteor, color="#1abc9c")
plt.xlabel("Alignment Agreement")
plt.ylabel("METEOR")
plt.title("Alignment Agreement vs METEOR")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "agreement_vs_meteor.png"), dpi=300)
plt.close()

# Plot 5: Correlation Heatmap
# We'll create a correlation matrix using the three metrics: Entropy, Agreement, METEOR
metrics_matrix = np.array([filtered_entropy, filtered_agreement, filtered_meteor])
corr_matrix = np.corrcoef(metrics_matrix)
labels = ["Entropy", "Agreement", "METEOR"]

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="coolwarm", cbar=False)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
plt.close()

print(f"\nVisualizations saved in {output_dir}")
print("Figures generated:")
print("- attention_entropy_distribution.png")
print("- entropy_vs_meteor.png")
print("- entropy_vs_agreement.png")
print("- agreement_vs_meteor.png")
print("- correlation_matrix.png")

print("\nDone.")

# Set up visualization style
sns.set_theme(context="talk", style="white", font="sans-serif")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "axes.linewidth": 0.8
})

def plot_enhanced_attention_heatmap(attention_matrix, source_tokens, target_tokens, save_path):
    """Plot enhanced attention heatmap with different normalization schemes."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Enhanced Attention Analysis", fontsize=20)
    
    # Raw attention
    sns.heatmap(attention_matrix, xticklabels=source_tokens, yticklabels=target_tokens,
                cmap="Spectral", ax=axes[0,0], annot=True, fmt=".2f")
    axes[0,0].set_title("Raw Attention Weights")
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Row-normalized (per target token)
    row_norm = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
    sns.heatmap(row_norm, xticklabels=source_tokens, yticklabels=target_tokens,
                cmap="Spectral", ax=axes[0,1], annot=True, fmt=".2f")
    axes[0,1].set_title("Row-Normalized (Per Target Token)")
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Column-normalized (per source token)
    col_norm = attention_matrix / attention_matrix.sum(axis=0, keepdims=True)
    sns.heatmap(col_norm, xticklabels=source_tokens, yticklabels=target_tokens,
                cmap="Spectral", ax=axes[1,0], annot=True, fmt=".2f")
    axes[1,0].set_title("Column-Normalized (Per Source Token)")
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Softmax normalized
    softmax = np.exp(attention_matrix) / np.sum(np.exp(attention_matrix), axis=1, keepdims=True)
    sns.heatmap(softmax, xticklabels=source_tokens, yticklabels=target_tokens,
                cmap="Spectral", ax=axes[1,1], annot=True, fmt=".2f")
    axes[1,1].set_title("Softmax Normalized")
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_token_specific_analysis(attention_matrix, source_tokens, target_tokens, save_path, top_k=3):
    """Plot token-specific analysis showing top-k attended source tokens for each target token."""
    num_targets = min(len(target_tokens), 5)  # Limit to 5 target tokens for clarity
    
    fig, axes = plt.subplots(num_targets, 1, figsize=(12, 4*num_targets))
    if num_targets == 1:
        axes = [axes]
    
    for idx in range(num_targets):
        attn_weights = attention_matrix[idx]
        top_indices = np.argsort(attn_weights)[-top_k:][::-1]
        top_tokens = [source_tokens[i] for i in top_indices]
        top_weights = attn_weights[top_indices]
        
        sns.barplot(x=top_weights, y=top_tokens, palette="viridis", ax=axes[idx])
        axes[idx].set_title(f"Top-{top_k} attended source tokens for target token: {target_tokens[idx]}")
        axes[idx].set_xlabel("Attention Weight")
        axes[idx].set_ylabel("Source Token")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# After processing batches and before saving results, add visualization code:
print("\nGenerating visualizations...")

# Create output directory for visualizations
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
vis_dir = f"visualizations_{timestamp}"
os.makedirs(vis_dir, exist_ok=True)

# Process a sample of translations for visualization
num_samples = min(5, len(all_generated_texts))
for idx in range(num_samples):
    # Get attention matrix for this sample
    attn_matrix = all_attention_matrices[idx].mean(axis=0)  # Average over heads
    
    # Get source and target tokens using the model's tokenizer
    source_tokens = tokenizer.tokenize(all_source_texts[idx])
    target_tokens = tokenizer.tokenize(all_generated_texts[idx])
    
    # Ensure matrix and token dimensions match
    attn_matrix = attn_matrix[:len(target_tokens), :len(source_tokens)]
    
    # Generate visualizations
    plot_enhanced_attention_heatmap(
        attn_matrix,
        source_tokens,
        target_tokens,
        os.path.join(vis_dir, f"enhanced_attention_sample_{idx+1}.png")
    )
    
    plot_token_specific_analysis(
        attn_matrix,
        source_tokens,
        target_tokens,
        os.path.join(vis_dir, f"token_analysis_sample_{idx+1}.png")
    )

print(f"\nVisualizations saved in directory: {vis_dir}")