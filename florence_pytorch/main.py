from florence.configuration_florence2 import *
from florence.florence_attn import *
import florence.modeling_florence2 as flor2
from florence.processor import *
from PIL import Image
import csv
import pandas as pd
import torch
import json

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    
    flo_model, processor = flor2.load("BASE_FT", device)

    os.makedirs("detailed_caption_embeddings", exist_ok=True)

    df = pd.read_csv('detailed_captions.csv', header=None, names=['filepath', 'caption_data'], engine='python')

    for index, row in df.iterrows():
        filepath = row['filepath']
        caption_data = row['caption_data']

        # Process label
        label_tokens = processor.tokenizer(label)
        label_tensor = torch.tensor(label_tokens['input_ids']).to(device=flo_model.device)
        label_embedding = flo_model.get_input_embeddings()(label_tensor)

        label_save_path = os.path.join("eaf_label_embeddings", f"{label.replace('/', '_').replace(' ', '_')}.pt")
        torch.save({
            'label': label,
            'embedding': label_embedding.detach().cpu()
        }, label_save_path)

        print(f'Processed label: {label}')
        print(f'Label embedding shape: {label_embedding.shape}, meaning {label_embedding.shape[0]-2} tokens plus start/end tokens')
        
        # Process definition
        def_tokens = processor.tokenizer(definition)
        def_tensor = torch.tensor(def_tokens['input_ids']).to(device=flo_model.device)
        def_embedding = flo_model.get_input_embeddings()(def_tensor)

        def_save_path = os.path.join("eaf_definition_embeddings", f"{label.replace('/', '_').replace(' ', '_')}_definition.pt")
        torch.save({
            'label': label,
            'definition': definition,
            'embedding': def_embedding.detach().cpu()
        }, def_save_path)

        print(f'Processed definition for: {label}')
        print(f'Definition embedding shape: {def_embedding.shape}, meaning {def_embedding.shape[0]-2} tokens plus start/end tokens')
        print('-' * 50)

    labels = ['Individual Activity', 'Representing Content', 'Student Writing', 'Individual Technology', 'Student Location', 'Desks-Sitting', 'Student(s) Standing or Walking']
    definitions = ['']
    for label in sentence:
        tokens = processor.tokenizer(label)
        t = torch.tensor(tokens['input_ids']).to(device=flo_model.device)
        embedding = flo_model.get_input_embeddings()(t)
        save_path = os.path.join("eaf_label_embeddings", f"{label}.pt")
        torch.save({
            'label': label,
            'embedding': embedding.detach().cpu()
        }, save_path)
        print(f'The tokens are as follows: {embedding} of size {embedding.shape} this means there are {embedding.shape[0]-2} words with 2 tokens for start and end and an embedding size of {embedding.shape[1]} from florence')
    return 0

if __name__ == '__main__':
    main()