# inference_pipeline.py

import os
import torch
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig
import json
import cv2

# ------------------- Configuration -------------------
EXCEL_PATH = "TVSum processed.xlsx"
VIDEO_DIR = "video_data"
PROMPT_TEMPLATE_PATH = "prompt_template.txt"
PROTOTYPE_PATH = "q_vectors.pth"  # Dict[layer_idx] = prototype vector
STUDENT_MODEL_PATH = "/storage/tousin.akhter/Anas/RVS/checkpoints/student_model.ckpt"
STUDENT_MODEL_ID = "google/paligemma2-3b-pt-224"
EXIT_LAYERS = [4, 8, 12]
SIM_THRESHOLD = 0.8
MAX_GEN_TOKENS = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Utils -------------------
def extract_1fps_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(round(fps)) if fps > 0 else 1
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            frames.append(pil_image)
        idx += 1

    cap.release()
    return frames


def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def load_prototype_vectors(path, device):
    q_dict = torch.load(path, map_location=device)
    for k in q_dict:
        q_dict[k] = q_dict[k].to(device)
    return q_dict


def load_prompt_template(path):
    with open(path, "r") as f:
        return f.read().strip()


def format_prompt(template, row):
    return template.format(
        title=row["title"],
        transcript=row["transcript_processed"],
        audio=row["clubbed_audio_processed"]
    )


# ------------------- Early Exit Function -------------------
def early_exit_generate_multi(
    student_model,
    processor,
    input_ids,
    attention_mask,
    pixel_values,
    prototype_q_dict,
    exit_layers,
    tau,
    max_tokens,
):
    with torch.no_grad():
        encoder_outputs = student_model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )

        decoder_input_ids = torch.tensor(
            [[student_model.config.decoder_start_token_id]],
            device=input_ids.device,
        )

        outputs = student_model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        for layer_idx in exit_layers:
            hidden_i = outputs.hidden_states[layer_idx][:, -1, :]
            sim = cosine_similarity(hidden_i.squeeze(0), prototype_q_dict[layer_idx])
            if sim >= tau:
                generated_ids = student_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=max_tokens,
                )
                return processor.decode(generated_ids[0], skip_special_tokens=True), layer_idx

        # fallback: full decoding
        generated_ids = student_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_tokens,
        )
        return processor.decode(generated_ids[0], skip_special_tokens=True), None


# ------------------- Main Inference -------------------
def main():
    print("Loading model and processor...")
    processor = PaliGemmaProcessor.from_pretrained(STUDENT_MODEL_ID)

    # Load model with 4-bit quantization if available
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_type=torch.bfloat16)
    model = PaliGemmaForConditionalGeneration.from_pretrained(STUDENT_MODEL_ID, quantization_config=bnb_config).to(DEVICE)
    
    if os.path.isfile(STUDENT_MODEL_PATH):
        print(f"Loading fine-tuned weights from {STUDENT_MODEL_PATH}")
        ckpt = torch.load(STUDENT_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)

    model.eval()

    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    prototype_q = load_prototype_vectors(PROTOTYPE_PATH, DEVICE)
    df = pd.read_excel(EXCEL_PATH)

    all_results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_id = str(row["video_id"]).strip()
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            print(f"Warning: Missing video {video_path}")
            continue

        prompt = format_prompt(prompt_template, row)
        frames = extract_1fps_frames(video_path)

        for i, frame in enumerate(frames):
            inputs = processor(text=prompt, images=frame, return_tensors="pt").to(DEVICE)

            prediction, exit_layer = early_exit_generate_multi(
                student_model=model,
                processor=processor,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                prototype_q_dict=prototype_q,
                exit_layers=EXIT_LAYERS,
                tau=SIM_THRESHOLD,
                max_tokens=MAX_GEN_TOKENS,
            )

            all_results.append({
                "video_id": video_id,
                "frame_idx": i,
                "prediction": prediction,
                "exit_layer": exit_layer,
            })

    # Save results
    with open("vlm_predictions_with_exit.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved predictions to vlm_predictions_with_exit.json")


if __name__ == "__main__":
    main()
