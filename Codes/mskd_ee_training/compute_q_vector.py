import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import numpy as np
import pandas as pd

# === Configs ===
MODEL_PATH = "/storage/tousin.akhter/Anas/RVS/checkpoints/student_model.ckpt"
EXIT_LAYERS = [4, 8, 12]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model and processor ===
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_PATH, output_hidden_states=True).to(DEVICE)
model.eval()
processor = PaliGemmaProcessor.from_pretrained(MODEL_PATH)

# === Load your own dataset ===
# Must be a CSV with columns: prompt,image_path
# df = pd.read_csv("path/to/your_dataset.csv")  # Modify this path
print(f"Loaded {len(df)} examples.")

exit_vectors = {l: [] for l in EXIT_LAYERS}

# === Iterate through dataset ===
for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["prompt"]
    image_path = row["image_path"]

    if not os.path.exists(image_path):
        print(f"Skipping missing image: {image_path}")
        continue

    image = Image.open(image_path).convert("RGB")
    inputs = processor(prompt, images=image, return_tensors="pt").to(DEVICE)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=DEVICE)

    with torch.no_grad():
        outputs = model.model(
            input_ids=decoder_input_ids,
            encoder_outputs=model.get_encoder()(**inputs, return_dict=True),
            output_hidden_states=True,
            return_dict=True,
        )

    for layer_idx in EXIT_LAYERS:
        hidden = outputs.decoder_hidden_states[layer_idx]  # shape: [1, seq_len, hidden_dim]
        last_token = hidden[:, -1, :]                      # shape: [1, hidden_dim]
        exit_vectors[layer_idx].append(last_token.squeeze(0).cpu().numpy())

# === Compute and save q vectors ===
q_vectors = {l: np.stack(exit_vectors[l]).mean(axis=0) for l in EXIT_LAYERS}
torch.save(q_vectors, "q_vectors.pth")
print("Saved q_vectors.pth with layers:", list(q_vectors.keys()))

