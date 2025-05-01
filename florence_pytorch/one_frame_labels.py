from florence.configuration_florence2 import *
from florence.florence_attn import *
import florence.modeling_florence2 as flor2
from florence.processor import *
from utils import run_example
import csv
import cv2
from PIL import Image, ImageDraw, ImageFont 

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    return frame_rgb

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    flo_model, processor = flor2.load("BASE_FT", device)

    video_file = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/314hours Videos/110.006.2018_ELA2_Year2_20180205.mp4"
    output_csv = "output.csv"
    timestamps = [3000, 3250, 3500] #3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500]
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_number", "caption_result"])

        video_path = video_file
        for timestamp_ms in timestamps:
            try:
                image = get_frame_at_timestamp(video_path, timestamp_ms)
                task_prompt = "<MORE_DETAILED_CAPTION>"
                print("here")
                result = run_example(image, processor, flo_model, task_prompt)['<MORE_DETAILED_CAPTION>']
                print("result phrase:", result)  
                task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
                result_2 = result = run_example(image, processor, flo_model, task_prompt, text_input=result)
                image_number = f"{os.path.splitext(video_file)[0]}_{timestamp_ms}ms"
                writer.writerow([image_number, result_2])
            except Exception as e:
                print(f"Skipping {video_file} at {timestamp_ms}ms due to error: {e}")


    sentence = "A book is used or held by teacher or student"

    tokens = processor.tokenizer(sentence)
    t = torch.tensor(tokens['input_ids']).to(device=flo_model.device)
    embedding = flo_model.get_input_embeddings()(t)
    print(f'The tokens are as follows: {embedding} of size {embedding.shape} this means there are {embedding.shape[0]-2} words with 2 tokens for start and end and an embedding size of {embedding.shape[1]} from florence')
    return 0

if __name__ == '__main__':
    main()