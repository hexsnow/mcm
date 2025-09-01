#!/usr/bin/env python3
"""
Script to tokenize the train.csv file using pre-trained tokenizers.
This script uses the HuggingFace Transformers library with state-of-the-art tokenizers.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import argparse
import json
from typing import List, Dict, Any
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV file and return a DataFrame."""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def identify_text_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns that contain text data suitable for tokenization."""
    text_columns = []
    
    for col in df.columns:
        # Check if column contains string data
        if df[col].dtype == 'object':
            # Sample some values to see if they're text
            sample_values = df[col].dropna().head(100)
            if len(sample_values) > 0:
                # Check if most values are strings with reasonable length
                text_like = sample_values.astype(str).str.len().between(10, 10000).mean()
                if text_like > 0.5:  # More than 50% look like text
                    text_columns.append(col)
                    logger.info(f"Identified text column: {col}")
    
    return text_columns

def load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """Load a pre-trained tokenizer from HuggingFace."""
    try:
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "[PAD]"
        
        logger.info(f"Tokenizer loaded successfully. Vocabulary size: {tokenizer.vocab_size}")
        return tokenizer
        
    except Exception as e:
        logger.error(f"Error loading tokenizer {tokenizer_name}: {e}")
        raise

def tokenize_texts(tokenizer: AutoTokenizer, texts: List[str], max_length: int = 512, 
                   truncation: bool = True, padding: str = 'max_length') -> Dict[str, Any]:
    """Tokenize a list of texts using the pre-trained tokenizer."""
    logger.info(f"Tokenizing {len(texts)} texts with max length {max_length}")
    
    # Prepare texts for tokenization
    processed_texts = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        processed_texts.append(text.strip())
    
    # Tokenize in batches for efficiency
    batch_size = 1000
    all_input_ids = []
    all_attention_masks = []
    all_token_counts = []
    truncated_count = 0
    
    for i in range(0, len(processed_texts), batch_size):
        batch_texts = processed_texts[i:i + batch_size]
        
        # Tokenize batch
        encodings = tokenizer(
            batch_texts,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Convert to lists and store
        input_ids = encodings['input_ids'].tolist()
        attention_masks = encodings['attention_mask'].tolist()
        
        # Count original tokens (before truncation)
        for j, text in enumerate(batch_texts):
            original_encoding = tokenizer(text, truncation=False, return_tensors='pt')
            original_length = original_encoding['input_ids'].shape[1]
            
            if original_length > max_length:
                truncated_count += 1
            
            all_input_ids.append(input_ids[j])
            all_attention_masks.append(attention_masks[j])
            all_token_counts.append(original_length)
        
        if (i + batch_size) % 5000 == 0:
            logger.info(f"Processed {i + batch_size}/{len(processed_texts)} texts")
    
    tokenized_data = {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
        'token_count': all_token_counts,
        'truncated_count': truncated_count
    }
    
    logger.info(f"Tokenization completed. {truncated_count} texts were truncated.")
    return tokenized_data

def save_tokenized_data(tokenized_data: Dict[str, Any], output_dir: str, base_name: str) -> None:
    """Save tokenized data to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as numpy arrays
    np.save(output_path / f"{base_name}_input_ids.npy", np.array(tokenized_data['input_ids']))
    np.save(output_path / f"{base_name}_attention_mask.npy", np.array(tokenized_data['attention_mask']))
    np.save(output_path / f"{base_name}_token_count.npy", np.array(tokenized_data['token_count']))
    
    # Save metadata
    metadata = {
        'num_samples': len(tokenized_data['input_ids']),
        'max_length': len(tokenized_data['input_ids'][0]) if tokenized_data['input_ids'] else 0,
        'truncated_count': tokenized_data['truncated_count'],
        'vocab_size': len(tokenized_data['input_ids'][0]) if tokenized_data['input_ids'] else 0
    }
    
    with open(output_path / f"{base_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved tokenized data to {output_path}")

def get_recommended_tokenizers() -> List[str]:
    """Return a list of recommended tokenizers for different use cases."""
    return [
        "bert-base-uncased",           # Good for general text, smaller vocab
        "bert-base-cased",             # Case-sensitive BERT
        "distilbert-base-uncased",     # Faster, smaller BERT
        "roberta-base",                # Better performance than BERT
        "microsoft/DialoGPT-medium",   # Good for conversational text
        "gpt2",                        # GPT-2 tokenizer
        "t5-base",                     # T5 tokenizer, good for various tasks
        "microsoft/DialoGPT-small",    # Smaller, faster GPT
        "distilroberta-base"           # Faster RoBERTa
    ]

def main():
    parser = argparse.ArgumentParser(description='Tokenize train.csv file using pre-trained tokenizers')
    parser.add_argument('--input', default='data/train.csv', help='Input CSV file path')
    parser.add_argument('--output', default='data/tokenized', help='Output directory for tokenized data')
    parser.add_argument('--tokenizer', default='bert-base-uncased', help='Pre-trained tokenizer to use')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--no-truncation', action='store_true', help='Disable truncation')
    parser.add_argument('--padding', default='max_length', choices=['max_length', 'longest', 'do_not_pad'], 
                       help='Padding strategy')
    parser.add_argument('--list-tokenizers', action='store_true', help='List recommended tokenizers and exit')
    
    args = parser.parse_args()
    
    if args.list_tokenizers:
        print("Recommended tokenizers:")
        for i, tokenizer in enumerate(get_recommended_tokenizers(), 1):
            print(f"{i:2d}. {tokenizer}")
        return
    
    try:
        # Load data
        df = load_data(args.input)
        
        # Identify text columns
        text_columns = identify_text_columns(df)
        
        if not text_columns:
            logger.warning("No text columns identified. Please check your data.")
            return
        
        # Combine all text columns
        all_texts = []
        for col in text_columns:
            texts = df[col].dropna().astype(str).tolist()
            all_texts.extend(texts)
        
        logger.info(f"Total texts to tokenize: {len(all_texts)}")
        
        # Load pre-trained tokenizer
        tokenizer = load_tokenizer(args.tokenizer)
        
        # Tokenize the data
        tokenized_data = tokenize_texts(
            tokenizer=tokenizer,
            texts=all_texts,
            max_length=args.max_length,
            truncation=not args.no_truncation,
            padding=args.padding
        )
        
        # Save tokenized data
        save_tokenized_data(tokenized_data, args.output, "train")
        
        # Save tokenizer info
        tokenizer_info = {
            'name': args.tokenizer,
            'vocab_size': tokenizer.vocab_size,
            'model_max_length': tokenizer.model_max_length,
            'pad_token': tokenizer.pad_token,
            'cls_token': getattr(tokenizer, 'cls_token', None),
            'sep_token': getattr(tokenizer, 'sep_token', None),
            'unk_token': getattr(tokenizer, 'unk_token', None)
        }
        
        with open(Path(args.output) / "tokenizer_info.json", 'w') as f:
            json.dump(tokenizer_info, f, indent=2)
        
        # Print summary statistics
        logger.info("=== Tokenization Summary ===")
        logger.info(f"Tokenizer used: {args.tokenizer}")
        logger.info(f"Total texts processed: {len(all_texts)}")
        logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
        logger.info(f"Max sequence length: {args.max_length}")
        logger.info(f"Texts truncated: {tokenized_data['truncated_count']}")
        logger.info(f"Average tokens per text: {np.mean(tokenized_data['token_count']):.1f}")
        logger.info(f"Text columns used: {text_columns}")
        logger.info(f"Padding strategy: {args.padding}")
        
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        raise

if __name__ == "__main__":
    main()
