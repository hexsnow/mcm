#!/usr/bin/env python3
"""
Script to generate embeddings from student responses in train.csv.
This script uses sentence-transformers to create high-quality text embeddings.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

import argparse
import json
import pickle
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm
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

def initialize_embedding_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """Initialize the sentence transformer model for embeddings."""
    try:
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Model dimension: {model.get_sentence_embedding_dimension()}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading embedding model {model_name}: {e}")
        raise

def encode_texts_batch(model: SentenceTransformer, texts: List[str], 
                      batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    """
    Encode texts in batches for efficiency.
    
    Args:
        model: SentenceTransformer model
        texts: List of text strings to encode
        batch_size: Number of texts to process at once
        show_progress: Whether to show progress bar
    
    Returns:
        numpy array of embeddings with shape (num_texts, embedding_dim)
    """
    logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
    
    # Prepare texts for encoding
    processed_texts = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        processed_texts.append(text.strip())
    
    # Encode in batches
    embeddings = []
    
    if show_progress:
        iterator = tqdm(range(0, len(processed_texts), batch_size), 
                       desc="Generating embeddings")
    else:
        iterator = range(0, len(processed_texts), batch_size)
    
    for i in iterator:
        batch_texts = processed_texts[i:i + batch_size]
        
        # Skip empty batches
        if not batch_texts:
            continue
        
        # Encode batch
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    all_embeddings = np.vstack(embeddings)
    
    logger.info(f"Generated embeddings with shape: {all_embeddings.shape}")
    return all_embeddings

def process_student_responses(df: pd.DataFrame, model: SentenceTransformer, 
                            column_name: str = 'StudentExplanation',
                            batch_size: int = 32) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Process all student responses and generate embeddings.
    
    Args:
        df: DataFrame containing the data
        model: SentenceTransformer model
        column_name: Name of the column containing student responses
        batch_size: Batch size for processing
    
    Returns:
        Tuple of (embeddings, id_to_index_mapping)
    """
    logger.info(f"Processing student responses from column: {column_name}")
    
    # Check if column exists
    if column_name not in df.columns:
        available_columns = list(df.columns)
        logger.error(f"Column '{column_name}' not found. Available columns: {available_columns}")
        raise ValueError(f"Column '{column_name}' not found")
    
    # Filter out rows with missing responses
    valid_responses = df[column_name].dropna()
    logger.info(f"Found {len(valid_responses)} valid responses out of {len(df)} total rows")
    
    # Create id to index mapping
    id_to_index = {}
    texts = []
    
    for idx, (row_id, response) in enumerate(valid_responses.items()):
        id_to_index[str(row_id)] = idx
        texts.append(response)
    
    # Generate embeddings
    embeddings = encode_texts_batch(model, texts, batch_size)
    
    return embeddings, id_to_index

def save_embeddings(embeddings: np.ndarray, output_dir: str, base_name: str = "embeddings") -> None:
    """Save embeddings to .npy file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    embeddings_file = output_path / f"{base_name}.npy"
    np.save(embeddings_file, embeddings)
    
    logger.info(f"Saved embeddings to {embeddings_file}")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"File size: {embeddings_file.stat().st_size / (1024*1024):.2f} MB")

def save_id_mapping(id_to_index: Dict[str, int], output_dir: str, 
                   base_name: str = "id_mapping", format: str = "json") -> None:
    """Save id to index mapping to file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if format.lower() == "json":
        mapping_file = output_path / f"{base_name}.json"
        with open(mapping_file, 'w') as f:
            json.dump(id_to_index, f, indent=2)
    elif format.lower() == "pkl":
        mapping_file = output_path / f"{base_name}.pkl"
        with open(mapping_file, 'wb') as f:
            pickle.dump(id_to_index, f)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pkl'")
    
    logger.info(f"Saved id mapping to {mapping_file}")
    logger.info(f"Number of mappings: {len(id_to_index)}")

def get_recommended_models() -> List[str]:
    """Return a list of recommended embedding models."""
    return [
        "all-MiniLM-L6-v2",        # Fast, good quality, 384d
        "all-mpnet-base-v2",       # High quality, 768d
        "all-MiniLM-L12-v2",       # Better quality, 384d
        "multi-qa-MiniLM-L6-v2",   # Good for Q&A, 384d
        "paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual, 384d
        "all-distilroberta-v1",    # Good balance, 768d
        "sentence-t5-base",        # T5-based, 768d
        "text-embedding-ada-002"   # OpenAI-style, 1536d (if available)
    ]

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings from student responses')
    parser.add_argument('--input', default='data/train.csv', help='Input CSV file path')
    parser.add_argument('--output', default='data/embeddings', help='Output directory for embeddings')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence transformer model to use')
    parser.add_argument('--column', default='StudentExplanation', help='Column name containing student responses')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--mapping-format', default='json', choices=['json', 'pkl'], 
                       help='Format for id mapping file')
    parser.add_argument('--list-models', action='store_true', help='List recommended models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Recommended embedding models:")
        for i, model in enumerate(get_recommended_models(), 1):
            print(f"{i:2d}. {model}")
        return
    
    try:
        # Load data
        df = load_data(args.input)
        
        # Initialize embedding model
        model = initialize_embedding_model(args.model)
        
        # Process student responses
        embeddings, id_to_index = process_student_responses(
            df=df,
            model=model,
            column_name=args.column,
            batch_size=args.batch_size
        )
        
        # Save embeddings
        save_embeddings(embeddings, args.output, "student_embeddings")
        
        # Save id mapping
        save_id_mapping(id_to_index, args.output, "student_id_mapping", args.mapping_format)
        
        # Print summary statistics
        logger.info("=== Embedding Generation Summary ===")
        logger.info(f"Model used: {args.model}")
        logger.info(f"Column processed: {args.column}")
        logger.info(f"Total responses processed: {len(id_to_index)}")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"Batch size used: {args.batch_size}")
        logger.info(f"Output directory: {args.output}")
        
        # Calculate some statistics
        logger.info(f"Embedding statistics:")
        logger.info(f"  Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
        logger.info(f"  Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.4f}")
        logger.info(f"  Min norm: {np.min(np.linalg.norm(embeddings, axis=1)):.4f}")
        logger.info(f"  Max norm: {np.max(np.linalg.norm(embeddings, axis=1)):.4f}")
        
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise

if __name__ == "__main__":
    main()
