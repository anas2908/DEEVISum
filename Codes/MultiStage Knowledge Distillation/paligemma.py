from datasets import load_dataset
from transformers import AutoProcessor, PaliGemmaProcessor
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.utils.data import Dataset
from typing import Any, Dict
import random
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from huggingface_hub import HfApi
api = HfApi()
from PIL import Image 
import lightning as L
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np
import json


from torch.utils.data import Dataset
from typing import Any, List, Dict
import random
import json
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from transformers import PaliGemmaForConditionalGeneration
from huggingface_hub import HfApi

from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
api = HfApi()
from huggingface_hub import hf_hub_download

HF_TOKEN = "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"
file_path = hf_hub_download(repo_id="google/paligemma-3b-pt-224", filename="config.json", token=HF_TOKEN)

print(file_path)
MAX_LENGTH = 512
MODEL_ID = "google/paligemma-3b-pt-224"


class CustomDataset(Dataset):
    """
    PyTorch Dataset for paligemmaVa. Loads data from a JSON file.
    """
    def __init__(self, json_path: str):
        super().__init__()
        
        # Load JSON file
        with open(json_path, "r") as f:
            self.data = json.load(f)
        
    def __len__(self) -> int:
        return len(self.data)

    def format_target(self, gt: str) -> str:
        """
        Formats ground truth to match the structured token sequence format.
        """
        return f"<s_answer>{gt}</s_answer>"

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            Dict: A dictionary containing:
                - image (PIL.Image): The original Receipt image.
                - target_sequence (str): Tokenized ground truth sequence.
                - meta_description (str): Additional textual metadata.
        """
        sample = self.data[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        # image = Image.open(sample["image_path"])

        # Load ground truth as string
        # target_sequence = str(sample["gt"])
        target_sequence = self.format_target(sample["gt"])

        # Load meta description from file
        with open(sample["meta_description_path"], "r") as f:
            meta_description = f.read().strip()

        return image, target_sequence, meta_description


train_dataset = CustomDataset("dataset.json")
val_dataset = CustomDataset("dataset.json")






# processor = AutoProcessor.from_pretrained(MODEL_ID)
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)



# PROMPT = "extract JSON."

def train_collate_fn(examples):
  images = [example[0] for example in examples]
#   texts = [f"<image> {text}" for text in [example[2] for example in examples]]
  texts = [example[2] for example in examples]
  labels = [example[1] for example in examples]

  inputs = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding=True,
                     truncation="only_second", max_length=MAX_LENGTH,)

  input_ids = inputs["input_ids"]
  token_type_ids = inputs["token_type_ids"]
  attention_mask = inputs["attention_mask"]
  pixel_values = inputs["pixel_values"]
  labels = inputs["labels"]

  return input_ids, token_type_ids, attention_mask, pixel_values, labels


def eval_collate_fn(examples):
  images = [example[0] for example in examples]
  answers = [example[1] for example in examples]
#   texts = [f"<image> {text}" for text in [example[2] for example in examples]]
  texts = [example[2] for example in examples]

  inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]
  pixel_values = inputs["pixel_values"]

  return input_ids, attention_mask, pixel_values, answers



#  def train_collate_fn(examples):
#     images = []
#     texts = []
#     for example in examples:
#         image, ground_truth, meta_description = example
#         images.append(image)
#         # TODO: in the future we can replace this by processor.apply_chat_template
#         prompt = f"USER: <image>\n{meta_description}\nASSISTANT: {ground_truth}"
#         texts.append(prompt)


#     batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

#     labels = batch["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100
#     batch["labels"] = labels

#     input_ids = batch["input_ids"]
#     attention_mask = batch["attention_mask"]
#     pixel_values = batch["pixel_values"]
#     labels = batch["labels"]

#     return input_ids, attention_mask, pixel_values, labels


# def eval_collate_fn(examples):
#     # we only feed the prompt to the model
#     images = []
#     texts = []
#     answers = []
#     for example in examples:
#         image, ground_truth, meta_description = example
#         images.append(image)
#         # TODO: in the future we can replace this by processor.apply_chat_template
#         # prompt = f"USER: <image>\nExtract JSON.\nASSISTANT:"
#         prompt = f"USER: <image>\n{meta_description}\nASSISTANT:"
#         texts.append(prompt)
#         answers.append(ground_truth)

#     batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

#     input_ids = batch["input_ids"]
#     attention_mask = batch["attention_mask"]
#     pixel_values = batch["pixel_values"]

#     return input_ids, attention_mask, pixel_values, answers

  
train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=2, shuffle=True)
# input_ids, token_type_ids, attention_mask, pixel_values, labels = next(iter(train_dataloader))


val_dataloader = DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=2, shuffle=False)
# input_ids, attention_mask, pixel_values, answers = next(iter(val_dataloader))



class PaliGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch

        outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                pixel_values=pixel_values,
                                labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)



# use this for Q-LoRa
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map={"":0})
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
#trainable params: 11,298,816 || all params: 2,934,634,224 || trainable%: 0.38501616002417344







config = {"max_epochs": 3,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 2,
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True,
}

model_module = PaliGemmaModelPLModule(config, processor, model)



checkpoint_callback = ModelCheckpoint(
    dirpath="/home/tousin.akhter/Anas/RVS/chkpnts",  # Path to save the model
    filename="last_model",  # Name of the saved model file
    save_top_k=0,  # 0 ensures only the last checkpoint is saved
    save_last=True,  # Enables saving only the last model
)



trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback],
)

trainer.fit(model_module)