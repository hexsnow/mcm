#!/usr/bin/env python3
"""
Script to process and parse Category:Misconception labels from train.csv.
This script cleans and breaks down labels into topic and misconception parts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from typing import List, Dict, Any, Tuple, Optional
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_unique_labels(file_path: str = 'data/train.csv') -> pd.DataFrame:
    """Load unique Category:Misconception combinations from train.csv."""
    try:
        logger.info(f"Loading unique labels from {file_path}")
        df = pd.read_csv(file_path)
        
        # Get unique combinations of Category and Misconception
        unique_labels = df.groupby(['Category', 'Misconception']).size().reset_index(name='count')
        unique_labels = unique_labels.sort_values('count', ascending=False)
        
        logger.info(f"Found {len(unique_labels)} unique Category:Misconception combinations")
        logger.info(f"Total instances: {unique_labels['count'].sum()}")
        
        return unique_labels
        
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        raise

def parse_label(label: str) -> Dict[str, str]:
    """
    Parse a Category:Misconception label into topic and misconception parts.
    
    Args:
        label: Label in format "Category:Misconception" or similar
        
    Returns:
        Dictionary with 'topic' and 'misconception' keys
    """
    if pd.isna(label) or label == '':
        return {'topic': 'unknown', 'misconception': 'unknown'}
    
    label = str(label).strip()
    
    # Handle different label formats
    if ':' in label:
        # Format: "Category:Misconception"
        parts = label.split(':', 1)
        if len(parts) == 2:
            category, misconception = parts
            return {
                'topic': category.strip().lower(),
                'misconception': misconception.strip().lower()
            }
    
    # Handle format without colon (just Category or Misconception)
    if label.lower() in ['true_correct', 'false_correct', 'true_neither', 'false_neither']:
        return {
            'topic': 'correctness',
            'misconception': label.strip().lower()
        }
    elif label.lower() in ['true_misconception', 'false_misconception']:
        return {
            'topic': 'misconception',
            'misconception': label.strip().lower()
        }
    else:
        # Assume it's a misconception type
        return {
            'topic': 'mathematics',
            'misconception': label.strip().lower()
        }

def create_combined_label(category: str, misconception: str) -> str:
    """
    Create a combined label from Category and Misconception.
    
    Args:
        category: Category value
        misconception: Misconception value
        
    Returns:
        Combined label string
    """
    if pd.isna(misconception) or misconception == '':
        return f"{category}:NA"
    else:
        return f"{category}:{misconception}"

def process_labels(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process all unique labels and create structured data.
    
    Args:
        df: DataFrame with Category and Misconception columns
        
    Returns:
        List of processed label dictionaries
    """
    logger.info("Processing labels...")
    
    processed_labels = []
    
    for _, row in df.iterrows():
        category = row['Category']
        misconception = row['Misconception']
        count = row['count']
        
        # Create combined label
        combined_label = create_combined_label(category, misconception)
        
        # Parse the combined label
        parsed = parse_label(combined_label)
        
        # Create structured label data
        label_data = {
            'original_category': category,
            'original_misconception': misconception,
            'combined_label': combined_label,
            'topic': parsed['topic'],
            'misconception': parsed['misconception'],
            'count': int(count),
            'description': f"{parsed['topic']}: {parsed['misconception']}"
        }
        
        processed_labels.append(label_data)
    
    logger.info(f"Processed {len(processed_labels)} labels")
    return processed_labels

def save_labels_to_csv(processed_labels: List[Dict[str, Any]], output_path: str = 'data/labels.csv') -> None:
    """Save processed labels to CSV file."""
    try:
        df = pd.DataFrame(processed_labels)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(processed_labels)} labels to {output_path}")
    except Exception as e:
        logger.error(f"Error saving labels: {e}")
        raise

def save_labels_to_json(processed_labels: List[Dict[str, Any]], output_path: str = 'data/labels.json') -> None:
    """Save processed labels to JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(processed_labels, f, indent=2)
        logger.info(f"Saved {len(processed_labels)} labels to {output_path}")
    except Exception as e:
        logger.error(f"Error saving labels: {e}")
        raise

def prepare_texts_for_encoding(processed_labels: List[Dict[str, Any]]) -> List[str]:
    """
    Prepare final list of texts for encoding.
    
    Args:
        processed_labels: List of processed label dictionaries
        
    Returns:
        List of text strings ready for encoding
    """
    texts = []
    
    for label in processed_labels:
        # Add different text representations
        texts.append(label['combined_label'])
        texts.append(label['description'])
        texts.append(f"{label['topic']} - {label['misconception']}")
        
        # Add original values if they exist
        if label['original_category']:
            texts.append(str(label['original_category']))
        if label['original_misconception']:
            texts.append(str(label['original_misconception']))
    
    # Remove duplicates and empty strings
    texts = list(set([text for text in texts if text and text.strip()]))
    
    logger.info(f"Prepared {len(texts)} unique texts for encoding")
    return texts

def get_label_statistics(processed_labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics about the processed labels."""
    stats = {
        'total_labels': len(processed_labels),
        'total_instances': sum(label['count'] for label in processed_labels),
        'unique_topics': len(set(label['topic'] for label in processed_labels)),
        'unique_misconceptions': len(set(label['misconception'] for label in processed_labels)),
        'top_topics': {},
        'top_misconceptions': {}
    }
    
    # Count topics
    topic_counts = {}
    for label in processed_labels:
        topic = label['topic']
        topic_counts[topic] = topic_counts.get(topic, 0) + label['count']
    
    stats['top_topics'] = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Count misconceptions
    misconception_counts = {}
    for label in processed_labels:
        misconception = label['misconception']
        misconception_counts[misconception] = misconception_counts.get(misconception, 0) + label['count']
    
    stats['top_misconceptions'] = dict(sorted(misconception_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Process Category:Misconception labels')
    parser.add_argument('--input', default='data/train.csv', help='Input CSV file path')
    parser.add_argument('--output-csv', default='data/labels.csv', help='Output CSV file path')
    parser.add_argument('--output-json', default='data/labels.json', help='Output JSON file path')
    parser.add_argument('--show-stats', action='store_true', help='Show label statistics')
    parser.add_argument('--show-samples', action='store_true', help='Show sample parsed labels')
    
    args = parser.parse_args()
    
    try:
        # Load unique labels
        unique_labels = load_unique_labels(args.input)
        
        # Process labels
        processed_labels = process_labels(unique_labels)
        
        # Save to files
        save_labels_to_csv(processed_labels, args.output_csv)
        save_labels_to_json(processed_labels, args.output_json)
        
        # Prepare texts for encoding
        texts_for_encoding = prepare_texts_for_encoding(processed_labels)
        
        # Get statistics
        stats = get_label_statistics(processed_labels)
        
        # Print summary
        logger.info("=== Label Processing Summary ===")
        logger.info(f"Total unique labels: {stats['total_labels']}")
        logger.info(f"Total instances: {stats['total_instances']}")
        logger.info(f"Unique topics: {stats['unique_topics']}")
        logger.info(f"Unique misconceptions: {stats['unique_misconceptions']}")
        logger.info(f"Texts prepared for encoding: {len(texts_for_encoding)}")
        
        if args.show_stats:
            logger.info("\n=== Top Topics ===")
            for topic, count in stats['top_topics'].items():
                logger.info(f"  {topic}: {count}")
            
            logger.info("\n=== Top Misconceptions ===")
            for misconception, count in stats['top_misconceptions'].items():
                logger.info(f"  {misconception}: {count}")
        
        if args.show_samples:
            logger.info("\n=== Sample Parsed Labels ===")
            for i, label in enumerate(processed_labels[:5]):
                logger.info(f"  {i+1}. {label['combined_label']} → topic: {label['topic']}, misconception: {label['misconception']}")
        
        # Save texts for encoding
        texts_file = Path(args.output_csv).parent / 'texts_for_encoding.txt'
        with open(texts_file, 'w') as f:
            for text in texts_for_encoding:
                f.write(text + '\n')
        logger.info(f"Saved texts for encoding to {texts_file}")
        
    except Exception as e:
        logger.error(f"Error during label processing: {e}")
        raise

if __name__ == "__main__":
    main()
