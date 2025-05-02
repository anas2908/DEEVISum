# inference_early_exit.py

import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import argparse
import torch.nn.functional as F
import os


def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def early_exit_generate_multi(
    student_model,
    processor,
    input_ids,
    attention_mask,
    pixel_values,
    prototype_q_dict,  # dict: layer_idx → q vector
    exit_layers=[4, 8, 12],
    tau=0.8,
    max_tokens=256,
    use_early_exit=True,
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
            device=input_ids.device
        )

        if use_early_exit:
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

        # If early exit is disabled or no early exit condition met
        generated_ids = student_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_tokens,
        )
        return processor.decode(generated_ids[0], skip_special_tokens=True), None

def load_inputs(processor, prompt, image_path, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    return inputs


def load_prototype(path_or_tensor, device):
    if isinstance(path_or_tensor, str):
        return torch.load(path_or_tensor, map_location=device)
    return path_or_tensor.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_id', type=str, required=True, help='Huggingface model ID')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--prototype', type=str, required=True, help='Path to saved prototype vector (torch .pt file)')
    parser.add_argument('--tau', type=float, default=0.8, help='Threshold for cosine similarity')
    parser.add_argument('--max_tokens', type=int, default=256, help='Maximum tokens to generate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load processor & model
    processor = PaliGemmaProcessor.from_pretrained(args.model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(args.model_id)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    model.to(device)
    model.eval()

    # Prepare input
    inputs = load_inputs(processor, args.prompt, args.image, device)

    # Load prototype vector
    prototype_q = load_prototype(args.prototype, device)

    # Run early exit inference
    prediction, used_early_exit = early_exit_generate(
        student_model=model,
        processor=processor,
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pixel_values=inputs['pixel_values'],
        prototype_q=prototype_q,
        tau=args.tau,
        max_tokens=args.max_tokens,
    )

    print(f"Prediction: {prediction}")
    print(f"Used Early Exit: {used_early_exit}")


if __name__ == '__main__':
    main()

