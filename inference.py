import torch
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import argparse
import os

class MyPipeline(StableDiffusionPipeline):
    def run_safety_checker(self, image, device, dtype):
        return image, None


def inference_indomain(model_path, save_path):
    pipe = MyPipeline.from_pretrained(model_path, torch_dtype=torch.float16, equires_safety_checker=False)
    pipe.to("cuda")
    os.makedirs(f"{save_path}/indomain", exist_ok=True)
    for i in range(8):
        dataset = load_dataset("lambdalabs/naruto-blip-captions")
        item = dataset['train'][i]
        text = item['text']
        image = pipe(prompt=text).images[0]
        image.save(f"{save_path}/indomain/{text}.png")

def inference_outdomain(model_path, save_path):
    pipe = MyPipeline.from_pretrained(model_path, torch_dtype=torch.float16, requires_safety_checker=False)
    pipe.register_to_config(requires_safety_checker=False)
    pipe.to("cuda")
    os.makedirs(f"{save_path}/outdomain", exist_ok=True)
    prompts = [
        "A cat is sitting on a chair",
        "A dog is playing with a ball",
        "A person is riding a bike",
        "A person is playing guitar",
        "A person is playing a video game",
        "A person is playing soccer",
        "A person is playing basketball",
        "A person is playing tennis",
    ]
    for text in prompts:
        image = pipe(prompt=text).images[0]
        image.save(f"{save_path}/outdomain/{text}.png")
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    inference_indomain(args.model_path, args.output)
    inference_outdomain(args.model_path, args.output)