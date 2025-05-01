import cv2

from transformers import AutoProcessor, AutoModelForCausalLM
import requests
import torch

from PIL import Image, ImageDraw, ImageFont 
import random
import numpy as np
import copy
import os
import csv

def get_frame_at_timestamp(video_path, timestamp_ms):
    print("Getting frame")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not retrieve frame at {timestamp_ms}ms from {video_path}")

    # Convert BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("got frame")
    return Image.fromarray(frame_rgb)

def run_example(task_prompt, image, text_input=None):
    print("running example")
    model_id = "microsoft/Florence-2-base-ft"

    # Load model and processor
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

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

    print("ran example")

    return parsed_answer


video_file = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/314hours Videos/110.006.2018_ELA2_Year2_20180205.mp4"
output_csv = "output.csv"
timestamps = [3000, 3250, 3500] #3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500]
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_number", "caption_result"])

    # for video_file in os.listdir(video_folder):
    #     if not video_file.endswith(".mp4"):
    #         continue

    # video_path = os.path.join(video_folder, video_file)
    video_path = video_file
    for timestamp_ms in timestamps:
        try:
            image = get_frame_at_timestamp(video_path, timestamp_ms)
            task_prompt = "<MORE_DETAILED_CAPTION>"
            result = run_example(task_prompt, image)['<MORE_DETAILED_CAPTION>']
            print("result phrase:", result)  
            task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
            result_2 = run_example(task_prompt, image, text_input=result)
            image_number = f"{os.path.splitext(video_file)[0]}_{timestamp_ms}ms"
            writer.writerow([image_number, result_2])
        except Exception as e:
            print(f"Skipping {video_file} at {timestamp_ms}ms due to error: {e}")

# def main():
#     video_folder = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/314hours Videos/110.006.2018_ELA2_Year2_20180205.mp4"
#     output_csv = "output.csv"
#     timestamps = [1000, 1500, 2000]  # Replace with your desired list of timestamps (in ms)

#     get_captions(video_folder, output_csv, timestamps)

# if __name__ == "__main__":
#     main()
