import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import os
from datetime import datetime

def setup_wandb():
    """Initialize wandb run"""
    # Set up wandb API key
    wandb.login(key="6f56843a9fda5209d74d64d2c778e60eddbf7ef2")
    
    return wandb.init(
        project="mt5-translation",
        config={
            "model_name": "google/mt5-large",
            "batch_size": 8,
            "learning_rate": 1e-4,
            "num_epochs": 20,
            "max_length": 64,
            "dataset": "wmt14",
            "dataset_size": 100000
        }
    )

def create_model_dir():
    """Create directory for saving models"""
    # Create base models directory
    base_dir = "models"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir

def collate_fn(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in input_ids], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in attention_mask], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in labels], batch_first=True, padding_value=-100)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def main():
    # Set up wandb API key
    wandb.login(key="6f56843a9fda5209d74d64d2c778e60eddbf7ef2")
    
    # Initialize wandb
    run = setup_wandb()
    
    # Create model directory
    model_dir = create_model_dir()
    print(f"Saving models to: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = wandb.config.model_name
    model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset("wmt14", "de-en", split=f"train[:{wandb.config.dataset_size}]")

    def preprocess_data(examples):
        inputs = [ex['en'] for ex in examples['translation']]
        targets = [ex['de'] for ex in examples['translation']]
        
        model_inputs = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=wandb.config.max_length,
            return_tensors=None
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                padding=True,
                truncation=True,
                max_length=wandb.config.max_length,
                return_tensors=None
            )
        
        return {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': labels['input_ids']
        }

    # Process dataset
    dataset = dataset.map(preprocess_data, batched=True, batch_size=wandb.config.batch_size)
    dataset.set_format(type='torch')
    dataloader = DataLoader(
        dataset, 
        batch_size=wandb.config.batch_size, 
        shuffle=True, 
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    # Training setup
    optimizer = AdamW(model.parameters(), lr=wandb.config.learning_rate)
    best_loss = float('inf')
    
    # Training loop
    model.train()
    for epoch in range(wandb.config.num_epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            # Update metrics
            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1
            
            # Update progress bar and wandb
            current_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
            wandb.log({
                "batch_loss": loss_value,
                "running_loss": current_loss,
                "epoch": epoch + 1
            })
        
        # Calculate epoch statistics
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch + 1} - Average loss: {avg_loss:.4f}")
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch_loss": avg_loss,
            "epoch": epoch + 1
        })
        
        # Save model if it's the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best loss! Saving model...")
            model_path = os.path.join(model_dir, f"best_model_epoch_{epoch + 1}")
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            # Log model to wandb
            artifact = wandb.Artifact(
                name=f"model-epoch-{epoch + 1}", 
                type="model",
                description=f"Model checkpoint from epoch {epoch + 1} with loss {avg_loss:.4f}"
            )
            artifact.add_dir(model_path)
            run.log_artifact(artifact)
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Log final model to wandb
    final_artifact = wandb.Artifact(
        name="final_model", 
        type="model",
        description=f"Final model after training with best loss {best_loss:.4f}"
    )
    final_artifact.add_dir(final_model_path)
    run.log_artifact(final_artifact)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 