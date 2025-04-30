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
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
import numpy as np
import re

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

import json
import re
import numpy as np
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score

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
        # return f"<s_answer>{gt}</s_answer>"
        return gt

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


  
# train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=2, shuffle=True)
# input_ids, token_type_ids, attention_mask, pixel_values, labels = next(iter(train_dataloader))


# val_dataloader = DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=2, shuffle=False)
# input_ids, attention_mask, pixel_values, answers = next(iter(val_dataloader))



# class KnowledgeDistillationVLM(L.LightningModule):
#     def __init__(self, config, processor, student_model, teacher_model, train_dataset, val_dataset):
#         super().__init__()
#         self.config = config
#         self.processor = processor
#         self.student_model = student_model
#         self.teacher_model = teacher_model
#         self.teacher_model.eval()  # Teacher model remains in eval mode
        
#         # Store datasets
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset, 
#             batch_size=self.config["batch_size"], 
#             shuffle=True, 
#             collate_fn=train_collate_fn, 
#             num_workers=4
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset, 
#             batch_size=self.config["batch_size"], 
#             shuffle=False, 
#             collate_fn=eval_collate_fn, 
#             num_workers=4
#         )

#     def training_step(self, batch, batch_idx):
#         input_ids, token_type_ids, attention_mask, pixel_values, labels = batch
        
#         # Student Model Forward Pass
#         student_outputs = self.student_model(
#             input_ids=input_ids, attention_mask=attention_mask,
#             token_type_ids=token_type_ids, pixel_values=pixel_values, labels=labels
#         )
#         student_loss = student_outputs.loss
        
#         with torch.no_grad():
#             teacher_outputs = self.teacher_model(
#                 input_ids=input_ids, attention_mask=attention_mask,
#                 token_type_ids=token_type_ids, pixel_values=pixel_values
#             )

#         # Knowledge Distillation Loss (KL Divergence)
#         kd_loss = F.kl_div(
#             F.log_softmax(student_outputs.logits / 2.0, dim=-1),
#             F.softmax(teacher_outputs.logits / 2.0, dim=-1),
#             reduction="batchmean"
#         )
        
#         loss = student_loss + kd_loss * 0.5  # Weighted loss combination
#         self.log("train_loss", loss)

#         return loss

#     def validation_step(self, batch, batch_idx, dataset_idx=0):
#         input_ids, attention_mask, pixel_values, answers = batch
        
#         generated_ids = self.student_model.generate(
#             input_ids=input_ids, attention_mask=attention_mask,
#             pixel_values=pixel_values, max_new_tokens=MAX_LENGTH
#         )
#         predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        
#         scores = [
#             edit_distance(re.sub(r"(?:(?<=>) | (?=</s_))", "", pred), answer) / max(len(pred), len(answer))
#             for pred, answer in zip(predictions, answers)
#         ]
        
#         self.log("val_edit_distance", np.mean(scores))
#         return scores

#     def configure_optimizers(self):
#         return torch.optim.AdamW(self.student_model.parameters(), lr=self.config.get("lr"))

class KnowledgeDistillationVLM(L.LightningModule):
    def __init__(self, config, processor, student_model, mentor_model, teacher_model, train_dataset, val_dataset):
        super().__init__()
        self.config = config
        self.processor = processor

        # Models
        self.student_model = student_model
        self.mentor_model = mentor_model
        self.teacher_model = teacher_model

        # Teacher remains in evaluation mode
        # self.teacher_model.eval()

        # Enable manual optimization for multiple optimizers
        self.automatic_optimization = False
        
        # Learnable KD weights & temperature
        self.kd_weight_mentor = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))  # For Teacher → Mentor
        self.kd_weight_student = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))  # For Mentor → Student
        self.kd_weight_long= torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))  # For Teacher → Student
        self.T = config.get("temperature", 2.0)  # Default temperature

        # Store datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True, 
            collate_fn=train_collate_fn, 
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=False, 
            collate_fn=eval_collate_fn, 
            num_workers=4
        )

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch

        # ===== Teacher Model Forward Pass (Does not require gradient) =====
        # with torch.no_grad():
        #     teacher_outputs = self.teacher_model(
        #         input_ids=input_ids, attention_mask=attention_mask,
        #         token_type_ids=token_type_ids, pixel_values=pixel_values
        #     )
        #     teacher_logits = teacher_outputs.logits  # Teacher predictions

        teacher_outputs = self.teacher_model(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, pixel_values=pixel_values, labels=labels
        )
        teacher_logits = teacher_outputs.logits
        teacher_loss = teacher_outputs.loss  # teacher's own loss

        # ===== Mentor Model Forward Pass =====
        mentor_outputs = self.mentor_model(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, pixel_values=pixel_values, labels=labels
        )
        mentor_logits = mentor_outputs.logits
        mentor_loss = mentor_outputs.loss  # Mentor's own loss

        # ===== Student Model Forward Pass =====
        student_outputs = self.student_model(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, pixel_values=pixel_values, labels=labels
        )
        student_logits = student_outputs.logits
        student_loss = student_outputs.loss  # Student's own loss

        # ===== Knowledge Distillation Losses =====
        kd_loss_mentor = F.kl_div(
            F.log_softmax(mentor_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction="batchmean"
        ) * (self.T ** 2)

        kd_loss_student = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(mentor_logits / self.T, dim=-1),
            reduction="batchmean"
        ) * (self.T ** 2)

        kd_loss_long = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction="batchmean"
        ) * (self.T ** 2)

        # ===== Total Loss for Each Model =====
        total_teacher_loss = teacher_loss 
        total_mentor_loss = mentor_loss + kd_loss_mentor * self.kd_weight_mentor  # Mentor learns from teacher
        total_student_loss = student_loss + kd_loss_student * self.kd_weight_student +  kd_loss_long * self.kd_weight_long  # Student learns from mentor and Teacher

        # ===== Backpropagation & Optimizer Steps =====
        optimizer_teacher, optimizer_mentor, optimizer_student = self.optimizers()


        
        # Teacher Update
        optimizer_teacher.zero_grad()
        self.manual_backward(total_teacher_loss, retain_graph=True)
        # self.manual_backward(teacher_loss)  # <--- Ensure this is called before step()
        optimizer_teacher.step()


        # Forward Pass for Mentor (detach teacher's outputs)
        # with torch.no_grad():
        #     teacher_outputs = self.teacher_model(**batch)  # Ensure no gradients are stored


        # self.manual_backward(mentor_loss)
        # Mentor Update
        optimizer_mentor.zero_grad()
        self.manual_backward(total_mentor_loss, retain_graph=True)
        optimizer_mentor.step()

        # Forward Pass for Student (detach mentor's outputs)
        # with torch.no_grad():
        #     mentor_outputs = self.mentor_model(**batch)


        # self.manual_backward(student_loss)
        # Student Update
        optimizer_student.zero_grad()
        self.manual_backward(total_student_loss)
        optimizer_student.step()

        

        # Logging
        self.log("train_teacher_loss", total_teacher_loss)
        self.log("train_mentor_loss", total_mentor_loss)
        self.log("train_student_loss", total_student_loss)
        self.log("kd_weight_mentor", self.kd_weight_mentor)
        self.log("kd_weight_student", self.kd_weight_student)
        self.log("kd_weight_long", self.kd_weight_long)

        return total_student_loss  # Student loss is used for monitoring

    # def validation_step(self, batch, batch_idx, dataset_idx=0):
    #     input_ids, attention_mask, pixel_values, answers = batch
        
    #     generated_ids = self.student_model.generate(
    #         input_ids=input_ids, attention_mask=attention_mask,
    #         pixel_values=pixel_values, max_new_tokens=self.config.get("max_length", 256)
    #     )
    #     predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        
    #     scores = [
    #         edit_distance(re.sub(r"(?:(?<=>) | (?=</s_))", "", pred), answer) / max(len(pred), len(answer))
    #         for pred, answer in zip(predictions, answers)
    #     ]
        
    #     self.log("val_edit_distance", np.mean(scores))
    #     return scores

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, answers = batch
        
        generated_ids = self.student_model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, max_new_tokens=self.config.get("max_length", 256)
        )
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        
        rouge_l_scores = []
        bleu4_scores = []
        smoothing = SmoothingFunction().method1
        results = []  # Store prediction-answer pairs

        for pred, answer in zip(predictions, answers):
            # Clean predictions
            pred_clean = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer_clean = re.sub(r"(?:(?<=>) | (?=</s_))", "", answer)

            # ROUGE-L score calculation
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_l = scorer.score(answer_clean, pred_clean)['rougeL'].fmeasure
            rouge_l_scores.append(rouge_l)

            # BLEU-4 score calculation
            bleu4 = sentence_bleu([answer_clean.split()], pred_clean.split(), smoothing_function=smoothing)
            bleu4_scores.append(bleu4)

            # Store prediction-answer pair
            results.append({"prediction": pred_clean, "answer": answer_clean})

        # Logging average scores
        avg_rougeL = np.mean(rouge_l_scores)
        avg_bleu4 = np.mean(bleu4_scores)
        self.log("val_rougeL", avg_rougeL)
        self.log("val_bleu4", avg_bleu4)

        # Save JSON file with prediction-answer pairs
        if batch_idx == 0 and dataset_idx == 0:  # Save only once per epoch
            with open("predictions.json", "w") as f:
                json.dump(results, f, indent=4)

        # Return a dictionary with the scores
        return {"rougeL": avg_rougeL, "bleu4": avg_bleu4}


    def configure_optimizers(self):
        opt_teacher = torch.optim.AdamW(self.teacher_model.parameters(), lr=self.config.get("lr", 1e-4))
        opt_mentor = torch.optim.AdamW(self.mentor_model.parameters(), lr=self.config.get("lr", 1e-4))
        opt_student = torch.optim.AdamW(self.student_model.parameters(), lr=self.config.get("lr", 1e-4))
        return [opt_teacher, opt_mentor, opt_student]



# ===== Load Models =====
MODEL_ID_STUDENT = "google/paligemma2-3b-pt-224"
MODEL_ID_MENTOR = "google/paligemma2-10b-pt-224"  # New Mentor Model
MODEL_ID_TEACHER = "google/paligemma2-28b-pt-224"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_type=torch.bfloat16)
lora_config = LoraConfig(r=8, target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")

student_model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID_STUDENT, quantization_config=bnb_config, device_map={"": 0})
student_model = get_peft_model(student_model, lora_config)

mentor_model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID_MENTOR, quantization_config=bnb_config, device_map={"": 0})
mentor_model = get_peft_model(mentor_model, lora_config)

teacher_model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID_TEACHER, quantization_config=bnb_config, device_map={"": 0})
# teacher_model.eval()  # Teacher starts in evaluation mode
teacher_model = get_peft_model(teacher_model, lora_config)

# Training Configuration
config = {"max_epochs": 3, "gradient_clip_val": 1.0, "accumulate_grad_batches": 8, "lr": 1e-4, "batch_size": 2}
# config = {"max_epochs": 3, "gradient_clip_val": 1.0, "lr": 1e-4, "batch_size": 2}

model_module = KnowledgeDistillationVLM(
    config, processor, student_model, mentor_model, teacher_model, train_dataset, val_dataset
)

# Trainer
checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", filename="student_model", save_top_k=1, monitor="train_student_loss", mode="min")

trainer = L.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=config["max_epochs"],
    # accumulate_grad_batches=config["accumulate_grad_batches"],
    precision="16-mixed",
    callbacks=[checkpoint_callback],
)

trainer.fit(model_module) #72 GB GPU 