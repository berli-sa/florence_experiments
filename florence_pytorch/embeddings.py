import os
import csv
import json
import torch
from PIL import Image
import pandas as pd
from florence.configuration_florence2 import *
from florence.florence_attn import *
import florence.modeling_florence2 as flor2
from florence.processor import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
csv_path = "region_caption/book/0002.csv"
image_folder = "/standard/spencerNSF/NeuralNetworksProjectVideos/Data Science Competition/Bounding_McGee/frames/book/0002"
output_folder = "embeddings/book/0002"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(csv_path)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    
    flo_model, processor = flor2.load("BASE_FT", device)

    for idx, row in df.iterrows():
        image_num = f"{row['image_num']:04d}"
        caption_data = row['bboxes']

        image_path = os.path.join(image_folder, f"{image_num}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Could not load image {image_path}: {e}")
            continue

        caption_dict = eval(caption_data)

        bboxes = caption_dict['<DENSE_REGION_CAPTION>']['bboxes']
        labels = caption_dict['<DENSE_REGION_CAPTION>']['labels']

        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            x0, y0, x1, y1 = bbox

            cropped = image.crop((x0, y0, x1, y1))
            cropped_resized = cropped.resize((224, 224))

            img_tensor = processor(images=cropped_resized, return_tensors="pt").to(flo_model.device)
            img_embedding = flo_model.get_input_embeddings()(img_tensor['input_ids'])

            label_tokens = processor.tokenizer(text=label, return_tensors="pt")
            label_input_ids = label_tokens['input_ids'].to(device=flo_model.device)
            label_embedding = flo_model.get_input_embeddings()(label_input_ids)

            save_path = os.path.join(output_folder, f"{image_num}_{i}.pt")
            torch.save({
                'image_num': image_num,
                'bbox_index': i,
                'label': label,
                'cropped_image_embedding': img_embedding.detach().cpu(),
                'label_embedding': label_embedding.detach().cpu()
            }, save_path)

            print(f"Saved embedding for {image_num}, bbox {i}")


    print(f'The tokens are as follows: {img_embedding} of size {img_embedding.shape} this means there are {img_embedding.shape[1]-2} words with 2 tokens for start and end and an embedding size of {img_embedding.shape[2]} from florence')
    print(f'The tokens are as follows: {label_embedding} of size {label_embedding.shape} this means there are {label_embedding.shape[1]-2} words with 2 tokens for start and end and an embedding size of {label_embedding.shape[2]} from florence')
    print(label)
    return 0

if __name__ == '__main__':
    main()

