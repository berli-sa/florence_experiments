from transformers import AutoProcessor, AutoModelForCausalLM
print(AutoModelForCausalLM.__module__)
from PIL import Image
import requests
import torch

import matplotlib.pyplot as plt  
import matplotlib.patches as patches  

from PIL import Image, ImageDraw, ImageFont 
import random
import numpy as np
import copy
import os
import csv
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

# Define model ID
model_id = "microsoft/Florence-2-base-ft"

# Load model and processor
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def draw_polygons(image, prediction, fill_mask=False):  
    """  
    Draws segmentation masks with polygons on an image.  
  
    Parameters:  
    - image_path: Path to the image file.  
    - prediction: Dictionary containing 'polygons' and 'labels' keys.  
                  'polygons' is a list of lists, each containing vertices of a polygon.  
                  'labels' is a list of labels corresponding to each polygon.  
    - fill_mask: Boolean indicating whether to fill the polygons with color.  
    """  
    # Load the image  
   
    draw = ImageDraw.Draw(image)  
      
   
    # Set up scale factor if needed (use 1 if not scaling)  
    scale = 1  
      
    # Iterate over polygons and labels  
    for polygons, label in zip(prediction['polygons'], prediction['labels']):  
        color = random.choice(colormap)  
        fill_color = random.choice(colormap) if fill_mask else None  
          
        for _polygon in polygons:  
            _polygon = np.array(_polygon).reshape(-1, 2)  
            if len(_polygon) < 3:  
                print('Invalid polygon:', _polygon)  
                continue  
              
            _polygon = (_polygon * scale).reshape(-1).tolist()  
              
            # Draw the polygon  
            if fill_mask:  
                draw.polygon(_polygon, outline=color, fill=fill_color)  
            else:  
                draw.polygon(_polygon, outline=color)  
              
            # Draw the label text  
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  
  
    # Save or display the image  
    #image.show()  # Display the image  
    image.save("image4.png")

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    #print(f"inputs ---> {inputs}")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    #print(f"generated_ids ---> {generated_ids}")

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    #print(f"generated_text ---> {generated_text}")

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

def plot_bbox(image, data):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.savefig("image2.png")

book_dir = "/standard/spencerNSF/NeuralNetworksProjectVideos/Data Science Competition/Bounding_McGee/frames/book"
output_dir = "region_caption/book"

os.makedirs(output_dir, exist_ok=True)

for folder_name in sorted(os.listdir(book_dir)):
    folder_path = os.path.join(book_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue  # Skip files, only process folders

    output_csv_path = os.path.join(output_dir, f"{folder_name}.csv")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_num', 'bboxes', 'labels'])

        print(f"Processing folder {folder_name}...")

        for image_name in sorted(os.listdir(folder_path)):
            if not image_name.endswith('.jpg'):
                continue

            image_path = os.path.join(folder_path, image_name)

            try:
                image = Image.open(image_path)

                task_prompt = '<DENSE_REGION_CAPTION>'
                result = run_example(task_prompt)

                image_number = os.path.splitext(image_name)[0]

                writer.writerow([image_number, result])

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        