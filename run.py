"""
run.py - end-to-end Florence + EAF pipeline

Usage example:
python run.py \
  --video /path/to/video.mp4 \
  --eaf /path/to/annotations.eaf \
  --timestamps 3000 3250 3500 \
  --out_dir results \
  --region_csv region_caption/book/0002.csv \
  --region_image_folder /standard/.../frames/book/0002
"""

import argparse
import os, sys, csv, random, json, torch, cv2
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor, GPT2Tokenizer, GPT2LMHeadModel
import xml.etree.ElementTree as ET
from collections import defaultdict
from glob import glob

from florence.configuration_florence2 import *
from florence.florence_attn import *
import florence.modeling_florence2 as flor2
from florence.processor import *

def cross_similarity(t1: torch.Tensor, t2: torch.Tensor):
    """
    Compute a final cross-similarity score between two sets of token embeddings.
    t1: (n1, d) or (n1, d) torch tensor
    t2: (n2, d) torch tensor
    returns single float
    """
    t1_norm = t1 / (t1.norm(dim=-1, keepdim=True) + 1e-8)
    t2_norm = t2 / (t2.norm(dim=-1, keepdim=True) + 1e-8)
    sim_matrix = torch.matmul(t1_norm, t2_norm.T)  # (n1, n2)
    final_score = sim_matrix.max(dim=1)[0].mean()
    return final_score.item()

class NextWordModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device).eval()

    def next_word_distribution(self, sentence: str) -> torch.Tensor:
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab)
        next_token_logits = logits[:, -1, :]  # (batch, vocab)
        probs = F.softmax(next_token_logits, dim=-1)
        return probs.squeeze().cpu()

"""
EAF Parsing
"""

def format_label(tier_id: str) -> str:
    words = tier_id.strip().split()
    if not words:
        return ""
    return words[0].capitalize() + " " + " ".join(w.lower() for w in words[1:])

def parse_eaf_annotations(eaf_file: str) -> dict:
    """
    Parse .eaf and return {time_ms: [formatted_tier_label, ...], ...}
    """
    tree = ET.parse(eaf_file)
    root = tree.getroot()

    # TIME_ORDER -> TIME_SLOT elements
    time_order = root.find("TIME_ORDER")
    time_slots = {}
    for ts in time_order.findall("TIME_SLOT"):
        ts_id = ts.attrib["TIME_SLOT_ID"]
        ts_value = int(ts.attrib.get("TIME_VALUE", "0"))
        time_slots[ts_id] = ts_value

    annotations_per_frame = defaultdict(list)
    for tier in root.findall("TIER"):
        if "PARENT_REF" not in tier.attrib:
            continue
        tier_id = tier.attrib.get("TIER_ID", "unknown")
        formatted_tier_id = format_label(tier_id)
        for annotation in tier.findall(".//ALIGNABLE_ANNOTATION"):
            start_ts = annotation.attrib["TIME_SLOT_REF1"]
            end_ts = annotation.attrib["TIME_SLOT_REF2"]
            start_time = time_slots.get(start_ts, 0)
            end_time = time_slots.get(end_ts, 0)
            value_elt = annotation.find("ANNOTATION_VALUE")
            value = value_elt.text if value_elt is not None else ""
            if value and value.strip():
                # map entire interval -> append label to each ms in range (as you used previously)
                for t in range(start_time, end_time + 1):
                    annotations_per_frame[t].append(formatted_tier_id)

    return dict(annotations_per_frame)

"""
Video Frame Extraction
"""

def get_frame_at_timestamp(video_path: str, timestamp_ms: int) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not retrieve frame at {timestamp_ms}ms from {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


"""
Florence
"""

def load_florence(device: str = "cuda"):
    flo_model, processor = flor2.load("BASE_FT", device)
    return flo_model, processor

def run_florence_task(flo_model, processor, task_prompt: str, image: Image.Image, text_input: str = None):
    """
    Run Florence multimodal generation.
    Returns parsed post-processed result (dictionary keyed by task_prompt)
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # prepare inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

    # move to model device
    device = flo_model.device if hasattr(flo_model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # generation
    with torch.no_grad():
        generated_ids = flo_model.generate(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

"""
Embeddings
"""

def extract_region_and_label_embeddings(flo_model, processor, image: Image.Image, bboxes: list, labels: list, save_folder: str, image_num: str):
    """
    For each bbox/label: crop the image region, run processor and attempt to extract embeddings
    Saves .pt files to save_folder with naming image_num_i.pt
    Returns list of saved paths
    """
    os.makedirs(save_folder, exist_ok=True)
    saved_paths = []

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x0, y0, x1, y1 = bbox
        # ensure ints
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        cropped = image.crop((x0, y0, x1, y1)).convert("RGB")
        cropped_resized = cropped.resize((224, 224))

        # prepare processor inputs
        # NOTE: your embeddings.py used processor(images=..., return_tensors="pt") and then used 'input_ids' from it.
        img_tensor = processor(images=cropped_resized, return_tensors="pt")
        # If processor returned tokenized 'input_ids' (Florence sometimes tokenizes image tokens in multimodal mode)
        img_embedding = None
        if 'input_ids' in img_tensor:
            # move to device for model
            device = flo_model.device if hasattr(flo_model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            img_ids = img_tensor['input_ids'].to(device)
            with torch.no_grad():
                try:
                    img_embedding = flo_model.get_input_embeddings()(img_ids)
                except Exception as e:
                    print(f"[WARN] could not get input embeddings for image tokens: {e}")
                    img_embedding = None
        else:
            # If there are pixel_values, try a model forward to attempt to get representations
            if 'pixel_values' in img_tensor:
                device = flo_model.device if hasattr(flo_model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                pv = img_tensor['pixel_values'].to(device)
                with torch.no_grad():
                    # NOTE: depending on Florence version, you might be able to call flo_model.get_image_features or pass pixel_values to the model
                    # We'll attempt several common options. If none work, img_embedding will be left None.
                    try:
                        # try image encoder method
                        img_feats = None
                        if hasattr(flo_model, 'get_image_features'):
                            img_feats = flo_model.get_image_features(pixel_values=pv)
                        else:
                            # attempt a forward pass to get hidden states
                            out = flo_model(pixel_values=pv, output_hidden_states=True)
                            if hasattr(out, 'last_hidden_state'):
                                img_feats = out.last_hidden_state.mean(dim=1)
                            elif 'hidden_states' in out:
                                img_feats = out.hidden_states[-1].mean(dim=1)
                        if img_feats is not None:
                            img_embedding = img_feats.cpu()
                    except Exception as e:
                        print(f"[WARN] image forward/featurization attempt failed: {e}")
                        img_embedding = None

        # embeddings.py
        label_tokens = processor.tokenizer(text=label, return_tensors="pt")
        device = flo_model.device if hasattr(flo_model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        label_input_ids = label_tokens['input_ids'].to(device)
        with torch.no_grad():
            try:
                label_embedding = flo_model.get_input_embeddings()(label_input_ids).cpu()
            except Exception as e:
                print(f"[WARN] could not get label embedding via get_input_embeddings(): {e}")
                label_embedding = None

        save_path = os.path.join(save_folder, f"{image_num}_{i}.pt")
        torch.save({
            'image_num': image_num,
            'bbox_index': i,
            'label': label,
            'cropped_image_embedding': img_embedding.detach().cpu() if isinstance(img_embedding, torch.Tensor) else img_embedding,
            'label_embedding': label_embedding.detach().cpu() if isinstance(label_embedding, torch.Tensor) else label_embedding
        }, save_path)
        saved_paths.append(save_path)
        print(f"Saved embedding for {image_num}, bbox {i} -> {save_path}")

    return saved_paths

def extract_caption_embedding(flo_model, processor, image: Image.Image, caption_text: str, save_folder: str, image_num: str):
    """
    Create an embedding for the detailed caption text (and optionally image+text) and save to disk.
    Use flo_model.get_input_embeddings()
    and also try to get a joint image-text embedding if possible.
    """
    os.makedirs(save_folder, exist_ok=True)
    device = flo_model.device if hasattr(flo_model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # token embeddings for caption text
    label_tokens = processor.tokenizer(text=caption_text, return_tensors="pt")
    label_input_ids = label_tokens['input_ids'].to(device)
    with torch.no_grad():
        try:
            caption_token_embeddings = flo_model.get_input_embeddings()(label_input_ids).cpu()
        except Exception as e:
            print(f"[WARN] could not compute caption token embeddings via get_input_embeddings(): {e}")
            caption_token_embeddings = None

    # Try to get a joint image-text embedding if model supports it
    joint_embedding = None
    try:
        inputs = processor(text=caption_text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = flo_model(**{k: v for k,v in inputs.items() if k in ['input_ids','pixel_values']}, output_hidden_states=True)
            # attempt to build joint by averaging last hidden states if present
            if hasattr(out, 'last_hidden_state'):
                joint_embedding = out.last_hidden_state.mean(dim=1).cpu()
            elif 'hidden_states' in out:
                joint_embedding = out.hidden_states[-1].mean(dim=1).cpu()
    except Exception as e:
        print(f"[WARN] joint image-text embedding attempt failed: {e}")
        joint_embedding = None

    save_path = os.path.join(save_folder, f"{image_num}.pt")
    torch.save({
        'image_num': image_num,
        'caption': caption_text,
        'caption_token_embeddings': caption_token_embeddings,
        'joint_embedding': joint_embedding
    }, save_path)
    print(f"Saved caption embedding for {image_num} -> {save_path}")
    return save_path

"""
Visualizations
"""

COLORMAP = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red','lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def draw_polygons_on_image(image: Image.Image, prediction: dict, out_path: str, fill_mask=False):

    img = image.copy()
    draw = ImageDraw.Draw(img)
    if 'bboxes' in prediction and 'labels' in prediction:
        for bbox, label in zip(prediction['bboxes'], prediction['labels']):
            x0, y0, x1, y1 = map(int, bbox)
            color = random.choice(COLORMAP)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw.text((x0 + 4, y0 + 4), label, fill=color)
    elif 'polygons' in prediction and 'labels' in prediction:
        for polygons, label in zip(prediction['polygons'], prediction['labels']):
            color = random.choice(COLORMAP)
            for _polygon in polygons:
                _polygon = np.array(_polygon).reshape(-1, 2).tolist()
                draw.polygon(sum([_polygon], []), outline=color)
            if len(polygons) and len(polygons[0]):
                px, py = polygons[0][0][0], polygons[0][0][1]
                draw.text((px + 4, py + 4), label, fill=color)
    img.save(out_path)
    print(f"Saved annotated image {out_path}")

def save_heatmap(matrix_df: pd.DataFrame, out_path: str, title: str = "Similarity heatmap"):
    plt.figure(figsize=(max(6, matrix_df.shape[1] * 0.3), max(4, matrix_df.shape[0] * 0.3)))
    plt.imshow(matrix_df.astype(float), aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.yticks(range(len(matrix_df.index)), matrix_df.index)
    plt.xticks(range(len(matrix_df.columns)), matrix_df.columns, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved heatmap {out_path}")

"""
Main
"""

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    flo_model, processor = load_florence(device)
    next_word_model = NextWordModel(device=device)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 1) Parse EAF
    frame_annotations = parse_eaf_annotations(args.eaf)
    eaf_csv_path = os.path.join(out_dir, "eaf_labels.csv")
    with open(eaf_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "labels"])
        for t, labels in sorted(frame_annotations.items()):
            writer.writerow([t, json.dumps(labels)])
    print(f"Wrote EAF label CSV: {eaf_csv_path}")

    # 2) Timestamps (if provided use them, otherwise derive from EAF keys)
    if args.timestamps:
        timestamps = args.timestamps
    else:
        timestamps = sorted(frame_annotations.keys())[:100]  # default to first 100 if many
    print(f"Processing timestamps: {timestamps}")

    caption_embeddings_folder = os.path.join(out_dir, "detailed_caption_embeddings")
    os.makedirs(caption_embeddings_folder, exist_ok=True)
    region_embeddings_folder = os.path.join(out_dir, "region_embeddings")
    os.makedirs(region_embeddings_folder, exist_ok=True)
    annotated_images_folder = os.path.join(out_dir, "annotated_images")
    os.makedirs(annotated_images_folder, exist_ok=True)
    caption_csv_path = os.path.join(out_dir, "captions.csv")

    # 3) For each timestamp: extract frame -> run Florence captioning -> save caption + caption embedding
    with open(caption_csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["image_number", "timestamp_ms", "detailed_caption", "dense_region_result_path"])
        for ts in timestamps:
            try:
                image = get_frame_at_timestamp(args.video, ts)
            except Exception as e:
                print(f"[WARN] skipping timestamp {ts}: could not extract frame: {e}")
                continue

            task_prompt = "<MORE_DETAILED_CAPTION>"
            try:
                parsed = run_florence_task(flo_model, processor, task_prompt, image)
                detailed_caption = parsed.get(task_prompt, parsed) if isinstance(parsed, dict) else parsed
                if isinstance(detailed_caption, (list, dict)):
                    detailed_caption_text = json.dumps(detailed_caption)
                else:
                    detailed_caption_text = str(detailed_caption)
            except Exception as e:
                print(f"[WARN] Florence detailed caption failed for {ts}: {e}")
                detailed_caption_text = ""

            region_prompt = "<DENSE_REGION_CAPTION>"
            try:
                region_parsed = run_florence_task(flo_model, processor, region_prompt, image)
                region_result = region_parsed.get(region_prompt, region_parsed) if isinstance(region_parsed, dict) else region_parsed
            except Exception as e:
                print(f"[WARN] Florence dense region caption failed for {ts}: {e}")
                region_result = None

            region_save_path = os.path.join(out_dir, f"region_{ts}.json")
            with open(region_save_path, "w", encoding="utf-8") as rf:
                json.dump(region_result, rf, default=str)

            image_num = f"{os.path.splitext(os.path.basename(args.video))[0]}_{ts}ms"
            try:
                caption_embed_path = extract_caption_embedding(flo_model, processor, image, detailed_caption_text, caption_embeddings_folder, image_num)
            except Exception as e:
                print(f"[WARN] failed to extract caption embedding for {image_num}: {e}")
                caption_embed_path = None

            region_embeddings_saved = []
            try:
                rr = region_result
                if isinstance(rr, dict) and ('<DENSE_REGION_CAPTION>' in rr or 'bboxes' in rr):
                    if '<DENSE_REGION_CAPTION>' in rr:
                        rr_inner = rr['<DENSE_REGION_CAPTION>']
                    else:
                        rr_inner = rr
                    bboxes = rr_inner.get('bboxes', [])
                    labels = rr_inner.get('labels', [])
                    if len(bboxes) and len(labels):
                        region_embeddings_saved = extract_region_and_label_embeddings(flo_model, processor, image, bboxes, labels, region_embeddings_folder, image_num)
            except Exception as e:
                print(f"[WARN] failed to extract region embeddings for {image_num}: {e}")

            writer.writerow([image_num, ts, detailed_caption_text, region_save_path])

            if region_result and isinstance(region_result, dict):
                annotated_path = os.path.join(annotated_images_folder, f"annotated_{image_num}.png")
                try:
                    draw_polygons_on_image(image, rr_inner, annotated_path)
                except Exception as e:
                    print(f"[WARN] could not annotate or save image for {image_num}: {e}")

    print(f"Saved captions CSV: {caption_csv_path}")

    # 4) Similarity computations

    # eaf_definition_embeddings/*.pt
    label_embeddings_dir = "eaf_definition_embeddings"
    label_files = glob(os.path.join(label_embeddings_dir, "*.pt"))
    label_embeddings = {}
    for lf in label_files:
        try:
            d = torch.load(lf)
            label = d.get('label') or os.path.splitext(os.path.basename(lf))[0]
            label_embeddings[label] = {
                'definition': d.get('definition', None),
                'embedding': d.get('embedding', None)
            }
            print(f"Loaded label embedding: {label}")
        except Exception as e:
            print(f"[WARN] failed to load label file {lf}: {e}")

    caption_files = glob(os.path.join(caption_embeddings_folder, "*.pt"))
    caption_embeddings = {}
    for cf in caption_files:
        try:
            d = torch.load(cf)
            frame_key = os.path.splitext(os.path.basename(cf))[0]
            caption_embeddings[frame_key] = {
                'caption': d.get('caption', ''),
                'token_embeddings': d.get('caption_token_embeddings', None),
                'joint_embedding': d.get('joint_embedding', None)
            }
        except Exception as e:
            print(f"[WARN] failed to load caption embedding {cf}: {e}")

    target_labels = list(label_embeddings.keys())
    frames = sorted(caption_embeddings.keys())

    df_embeddings = pd.DataFrame(index=target_labels, columns=frames)
    df_text = pd.DataFrame(index=target_labels, columns=frames)

    for label in target_labels:
        label_entry = label_embeddings[label]
        label_emb = label_entry['embedding']
        label_def = label_entry['definition'] or label

        if label_emb is None:
            print(f"[WARN] label {label} missing embedding -> skipping")
            continue

        if isinstance(label_emb, np.ndarray):
            label_emb = torch.from_numpy(label_emb)

        if isinstance(label_emb, torch.Tensor) and label_emb.dim() == 3:
            label_tokens_emb = label_emb.squeeze(0)
        else:
            label_tokens_emb = label_emb

        for frame in frames:
            cap_entry = caption_embeddings[frame]
            caption_token_emb = cap_entry.get('token_embeddings')
            caption_text = cap_entry.get('caption', '')

            emb_sim = None
            text_sim = None

            try:
                if caption_token_emb is not None and label_tokens_emb is not None:
                    if isinstance(caption_token_emb, torch.Tensor) and caption_token_emb.dim() == 3:
                        cap_tokens = caption_token_emb.squeeze(0)
                    else:
                        cap_tokens = caption_token_emb
                    emb_sim = cross_similarity(label_tokens_emb, cap_tokens)
                elif caption_token_emb is None:
                    if cap_entry.get('joint_embedding') is not None:
                        je = cap_entry['joint_embedding']
                        if isinstance(je, torch.Tensor):
                            if je.dim() == 2:
                                emb_sim = cross_similarity(label_tokens_emb, je)
                            else:
                                emb_sim = cross_similarity(label_tokens_emb, je.squeeze(0))
            except Exception as e:
                print(f"[WARN] embedding similarity failed for label {label} vs frame {frame}: {e}")
                emb_sim = None

            try:
                p_label = next_word_model.next_word_distribution(label_def)
                p_caption = next_word_model.next_word_distribution(caption_text)
                text_sim = F.cosine_similarity(p_label, p_caption, dim=0).item()
            except Exception as e:
                print(f"[WARN] text similarity failed for label {label} vs frame {frame}: {e}")
                text_sim = None

            df_embeddings.loc[label, frame] = emb_sim
            df_text.loc[label, frame] = text_sim

    emb_csv = os.path.join(out_dir, "cross_similarity_heatmap.csv")
    text_csv = os.path.join(out_dir, "next_word_similarities_heatmap.csv")
    df_embeddings.to_csv(emb_csv)
    df_text.to_csv(text_csv)
    print(f"Saved embedding similarity CSV: {emb_csv}")
    print(f"Saved text similarity CSV: {text_csv}")

    df_embeddings_filled = df_embeddings.fillna(0).astype(float)
    df_text_filled = df_text.fillna(0).astype(float)
    save_heatmap(df_embeddings_filled, os.path.join(out_dir, "cross_similarity_heatmap.png"), title="Embedding Cross-Similarity")
    save_heatmap(df_text_filled, os.path.join(out_dir, "next_word_similarity_heatmap.png"), title="Next Word Similarity")

    print("\nTop matches by embedding similarity:")
    for label in df_embeddings.index:
        try:
            top = df_embeddings.loc[label].dropna().astype(float).nlargest(3)
            print(f"\n{label}:")
            for frame, score in top.items():
                cap_text = caption_embeddings[frame]['caption'][:120] if frame in caption_embeddings else ""
                print(f"  {frame}: {score:.4f} -> {cap_text}")
        except Exception as e:
            print(f"[WARN] could not compute top matches for {label}: {e}")

    print("\nPipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end Florence + EAF pipeline")
    parser.add_argument("--video", required=True, help="Path to video file (.mp4)")
    parser.add_argument("--eaf", required=True, help="Path to .eaf annotation file")
    parser.add_argument("--timestamps", nargs="+", type=int, help="List of timestamps in ms to process")
    parser.add_argument("--out_dir", default="results", help="Output directory")
    parser.add_argument("--region_csv", default="region_caption/book/0002.csv", help="CSV containing region captions (optional; used if provided)")
    parser.add_argument("--region_image_folder", default=None, help="Folder containing region images (only needed for region-based embedding extraction from precomputed CSVs)")
    args = parser.parse_args()