import torch
import os
import sys
import argparse
from PIL import Image
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from glob import glob
from torchvision import transforms
from torchvision.transforms.functional import pad

# Workaround for unnecessary flash_attn requirement
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

# Function to download and load the Florence-2 model
def download_and_load_model(model_name):
    print('Checking device...')
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f'Device available : {device}')
    model_path = os.path.join("models", model_name.replace('/', '_'))
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} model to: {model_path}")
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)
    print(f"Loading model {model_name}...")
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):  # Workaround for unnecessary flash_attn requirement
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded.")
    return model, processor

def load_image_paths_from_folder(folder_path):
    valid_image_extensions = ["jpg", "jpeg", "png"]
    image_paths = []
    for ext in valid_image_extensions:
        for image_path in glob(os.path.join(folder_path, f"*.{ext}")):
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            if not os.path.exists(txt_path):
                image_paths.append(image_path)
    return image_paths

def run_model_batch(image_paths, model, processor, task='caption', num_beams=3, max_new_tokens=1024, detail_mode=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prompts = {
        1: '<CAPTION>',
        2: '<DETAILED_CAPTION>',
        3: '<MORE_DETAILED_CAPTION>'
    }
    prompt = prompts.get(detail_mode, '<MORE_DETAILED_CAPTION>')
    inputs = {
        "input_ids": [],
        "pixel_values": []
    }
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert("RGB")
        input_data = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False)
        inputs["input_ids"].append(input_data["input_ids"])
        inputs["pixel_values"].append(input_data["pixel_values"])
        print(f"Processing image: {image_path}")
    inputs["input_ids"] = torch.cat(inputs["input_ids"]).to(device)
    inputs["pixel_values"] = torch.cat(inputs["pixel_values"]).to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
    )
    results = processor.batch_decode(generated_ids, skip_special_tokens=False)
    print(results)
    clean_results = [result.replace('</s>', '').replace('<s>', '').replace('<pad>', '') for result in results]
    return clean_results

def process_images_in_batches(folder_path, model, processor, batch_size=5):
    image_paths = load_image_paths_from_folder(folder_path)
    total_images = len(image_paths)
    
    for i in range(0, total_images, batch_size):
        batch_image_paths = image_paths[i:i + batch_size]
        captions = run_model_batch(batch_image_paths, model, processor, task='caption', detail_mode=3)
        
        for j, caption in enumerate(captions):
            ext = os.path.splitext(batch_image_paths[j])[1]
            with open(batch_image_paths[j].replace(ext, '.txt'), 'w') as caption_file:
                caption_file.write(caption)

def process_single_image(image_path, model, processor):
    captions = run_model_batch([image_path], model, processor, task='caption', detail_mode=3)
    
    if captions:
        ext = os.path.splitext(image_path)[1]
        with open(image_path.replace(ext, '.txt'), 'w') as caption_file:
            caption_file.write(captions[0])
        print(f"Caption generated for {image_path}")
    else:
        print(f"Failed to generate caption for {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for images using Florence-2 model.")
    parser.add_argument("--folder", help="Path to the folder containing images")
    parser.add_argument("--image", help="Path to a single image file")
    args = parser.parse_args()

    model_name = 'microsoft/Florence-2-large'
    model, processor = download_and_load_model(model_name)

    if args.image:
        process_single_image(args.image, model, processor)
    elif args.folder:
        process_images_in_batches(args.folder, model, processor, batch_size=1)
    else:
        print("Please provide either --folder or --image argument.")




