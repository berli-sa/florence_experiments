from similarities import cross_similarity
from next_word import next_word_distribution
import torch
import os
import numpy as np
import pandas as pd
from glob import glob
import re
import torch.nn.functional as F

def next_word_similarity(sentence1, sentence2):
    p1 = next_word_distribution(sentence1)
    p2 = next_word_distribution(sentence2)
    return F.cosine_similarity(p1, p2, dim=0)

def main():
    # Define the labels of interest
    target_labels = ['Individual activity', 'Student writing', 'Individual technology', 'Desks-sitting ', 'Student(s) standing or walking']
    
    # Load label definition embeddings
    label_embeddings = {}
    label_files = glob(os.path.join('eaf_definition_embeddings', '*.pt'))
    
    for label_file in label_files:
        data = torch.load(label_file)
        label = data['label']
        
        # Only process labels in our target list
        if label in target_labels:
            label_embeddings[label] = {
                'definition': data['definition'],
                'embedding': data['embedding']
            }
            print(f"Loaded label: {label}")
    
    # Load caption embeddings
    caption_embeddings = {}
    caption_files = glob(os.path.join('detailed_caption_embeddings', '*.pt'))
    
    for caption_file in caption_files:
        data = torch.load(caption_file)
        frame = os.path.basename(caption_file).replace('.pt', '')
        
        caption_embeddings[frame] = {
            'caption': data['caption'],
            'embedding': data['embedding']
        }
        print(f"Loaded caption for frame: {frame}")
    
    # Create a matrix to store similarities
    # Rows: Labels, Columns: Captions
    embedding_similarity_matrix = {}
    text_similarity_matrix = {}
    
    # Calculate similarities
    for label in target_labels:
        if label not in label_embeddings:
            print(f"Warning: Label '{label}' not found in embeddings")
            continue
            
        label_embed = label_embeddings[label]['embedding']
        label_text = label_embeddings[label]['definition']
        
        embedding_similarities = {}
        text_similarities = {}
        
        for frame, caption_data in caption_embeddings.items():
            caption_embed = caption_data['embedding']
            caption_text = caption_data['caption']

            emb_similarity = cross_similarity(label_embed, caption_embed)
            embedding_similarities[frame] = emb_similarity

            text_similarity = next_word_similarity(label_text, caption_text).item()
            text_similarities[frame] = text_similarity
            
        embedding_similarity_matrix[label] = embedding_similarities
        text_similarity_matrix[label] = text_similarities
        print(f"Computed similarities for label: {label}")
    
    # Convert to pandas DataFrame for easier analysis and saving
    frames = sorted(caption_embeddings.keys(), key=lambda x: int(x) if x.isdigit() else x)
    labels = [label for label in target_labels if label in label_embeddings]
    
    # Initialize the DataFrame with NaN values
    df_embedding = pd.DataFrame(index=labels, columns=frames)
    df_text = pd.DataFrame(index=labels, columns=frames)
    
    # Fill in the values
    for label in labels:
        for frame in frames:
            if frame in embedding_similarity_matrix[label]:
                df_embedding.loc[label, frame] = embedding_similarity_matrix[label][frame]
            if frame in text_similarity_matrix[label]:
                df_text.loc[label, frame] = text_similarity_matrix[label][frame]

    df_embedding.to_csv('embedding_similarities.csv')
    print(f"Saved embedding similarity matrix to 'embedding_similarities.csv'")
    
    df_text.to_csv('text_distribution_similarities.csv')
    print(f"Saved text distribution similarity matrix to 'text_distribution_similarities.csv'")
    
    print("\nEmbedding Similarity Statistics:")
    print(f"Average similarity: {df_embedding.values.mean():.4f}")
    print(f"Maximum similarity: {df_embedding.values.max():.4f}")
    print(f"Minimum similarity: {df_embedding.values.min():.4f}")
    
    print("\nText Distribution Similarity Statistics:")
    print(f"Average similarity: {df_text.values.mean():.4f}")
    print(f"Maximum similarity: {df_text.values.max():.4f}")
    print(f"Minimum similarity: {df_text.values.min():.4f}")
    
    # Find top matches for each label using embedding similarity
    print("\nTop caption matches for each label (embedding similarity):")
    for label in labels:
        top_matches = df_embedding.loc[label].nlargest(3)
        print(f"\n{label}:")
        for frame, score in top_matches.items():
            print(f"  Frame {frame}: {score:.4f} - {caption_embeddings[frame]['caption'][:100]}...")
    
    # Find top matches for each label using text distribution similarity
    print("\nTop caption matches for each label (text distribution similarity):")
    for label in labels:
        top_matches = df_text.loc[label].nlargest(3)
        print(f"\n{label}:")
        for frame, score in top_matches.items():
            print(f"  Frame {frame}: {score:.4f} - {caption_embeddings[frame]['caption'][:100]}...")
    return 0

if __name__ == "__main__":
    main()
