#!/usr/bin/env python
"""
Example client script for the Shuttle-Jaguar Modal API.
This script demonstrates how to interact with the deployed Modal API.

Make sure to deploy the API first:
    modal deploy jaguar_app.py
"""

import base64
import io
import json
import os
import requests
from PIL import Image
from typing import List, Optional, Dict, Any
import argparse

# Replace with your actual deployment URL after deploying
# The URL will look like: https://yourname--shuttle-jaguar--shuttlejaguarmodel-generate-api.modal.run
BASE_URL = "https://your-modal-deployment--shuttle-jaguar"

def generate_image(
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 3.5,
    steps: int = 4,
    max_seq_length: int = 256,
    seed: Optional[int] = None,
    save_path: str = "output.png",
    display: bool = False
) -> Dict[str, Any]:
    """
    Generate a single image using the Modal API.
    
    Args:
        prompt: Text prompt for image generation
        height: Image height in pixels
        width: Image width in pixels
        guidance_scale: Classifier-free guidance scale
        steps: Number of inference steps
        max_seq_length: Maximum sequence length for text encoder
        seed: Random seed for reproducibility
        save_path: Path to save the generated image
        display: Whether to display the image (requires display environment)
        
    Returns:
        Dict containing API response data
    """
    # Build URL with query parameters
    url = f"{BASE_URL}--shuttlejaguarmodel-generate-api.modal.run"
    params = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "max_seq_length": max_seq_length
    }
    
    if seed is not None:
        params["seed"] = seed
    
    print(f"Generating image for prompt: '{prompt}'")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"error": response.text}
    
    result = response.json()
    
    # Save the image
    if "image" in result:
        image_data = base64.b64decode(result["image"])
        image = Image.open(io.BytesIO(image_data))
        image.save(save_path)
        print(f"Image saved to {save_path}")
        
        # Display the image if requested
        if display:
            image.show()
    
    return result

def batch_generate(
    prompts: List[str],
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 3.5,
    steps: int = 4,
    max_seq_length: int = 256,
    base_seed: Optional[int] = None,
    output_dir: str = "outputs",
    display: bool = False
) -> Dict[str, Any]:
    """
    Generate multiple images in a batch using the Modal API.
    
    Args:
        prompts: List of text prompts
        height: Image height in pixels
        width: Image width in pixels
        guidance_scale: Classifier-free guidance scale
        steps: Number of inference steps
        max_seq_length: Maximum sequence length for text encoder
        base_seed: Base random seed (will be incremented for each image)
        output_dir: Directory to save generated images
        display: Whether to display the images (requires display environment)
        
    Returns:
        Dict containing API response data
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare request data
    url = f"{BASE_URL}--shuttlejaguarmodel-batch-api.modal.run"
    data = {
        "prompts": prompts,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "max_seq_length": max_seq_length
    }
    
    if base_seed is not None:
        data["base_seed"] = base_seed
    
    print(f"Batch generating {len(prompts)} images...")
    response = requests.post(url, json=data)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"error": response.text}
    
    result = response.json()
    
    # Save the images
    if "results" in result:
        for i, image_result in enumerate(result["results"]):
            if "image" in image_result:
                # Create a filename from the prompt
                prompt = image_result["prompt"]
                filename = f"{i+1:03d}_{prompt[:30].replace(' ', '_')}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save the image
                image_data = base64.b64decode(image_result["image"])
                image = Image.open(io.BytesIO(image_data))
                image.save(filepath)
                print(f"Image {i+1} saved to {filepath}")
                
                # Display the image if requested
                if display:
                    image.show()
    
    return result

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the model.
    
    Returns:
        Dict containing model information
    """
    url = f"{BASE_URL}--shuttlejaguarmodel-info.modal.run"
    
    print("Getting model information...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"error": response.text}
    
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="Client for Shuttle-Jaguar Modal API")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate image command
    gen_parser = subparsers.add_parser("generate", help="Generate a single image")
    gen_parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    gen_parser.add_argument("--height", type=int, default=1024, help="Image height")
    gen_parser.add_argument("--width", type=int, default=1024, help="Image width")
    gen_parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale")
    gen_parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    gen_parser.add_argument("--seed", type=int, help="Random seed")
    gen_parser.add_argument("--output", type=str, default="output.png", help="Output file path")
    gen_parser.add_argument("--display", action="store_true", help="Display the image")
    
    # Batch generate command
    batch_parser = subparsers.add_parser("batch", help="Generate multiple images")
    batch_parser.add_argument("--prompts-file", type=str, help="File containing prompts (one per line)")
    batch_parser.add_argument("--prompts", nargs="+", help="List of prompts")
    batch_parser.add_argument("--height", type=int, default=1024, help="Image height")
    batch_parser.add_argument("--width", type=int, default=1024, help="Image width")
    batch_parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale")
    batch_parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    batch_parser.add_argument("--base-seed", type=int, help="Base random seed")
    batch_parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    batch_parser.add_argument("--display", action="store_true", help="Display the images")
    
    # Info command
    subparsers.add_parser("info", help="Get model information")
    
    # Update URL command
    url_parser = subparsers.add_parser("set-url", help="Set the API base URL")
    url_parser.add_argument("url", type=str, help="Base URL for the API")
    
    args = parser.parse_args()
    
    # Check if BASE_URL is set to the default value
    global BASE_URL
    if BASE_URL == "https://your-modal-deployment--shuttle-jaguar" and args.command != "set-url":
        print("Warning: You need to set your actual Modal deployment URL first!")
        print("Run: python client_example.py set-url YOUR_DEPLOYMENT_URL")
        print("Or edit the BASE_URL variable in this script.")
        return
    
    if args.command == "generate":
        result = generate_image(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            seed=args.seed,
            save_path=args.output,
            display=args.display
        )
        print(f"Generation time: {result.get('generation_time', 'unknown')} seconds")
        
    elif args.command == "batch":
        prompts = []
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        elif args.prompts:
            prompts = args.prompts
        else:
            print("Error: Either --prompts-file or --prompts must be provided")
            return
            
        result = batch_generate(
            prompts=prompts,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            base_seed=args.base_seed,
            output_dir=args.output_dir,
            display=args.display
        )
        print(f"Total generation time: {result.get('total_generation_time', 'unknown')} seconds")
        
    elif args.command == "info":
        info = get_model_info()
        print("Model Information:")
        print(json.dumps(info, indent=2))
        
    elif args.command == "set-url":
        # Save the URL to a config file for persistence
        with open(os.path.join(os.path.dirname(__file__), '.api_url'), 'w') as f:
            f.write(args.url)
        print(f"API URL set to: {args.url}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    # Try to load saved URL if it exists
    config_path = os.path.join(os.path.dirname(__file__), '.api_url')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_url = f.read().strip()
            if saved_url:
                BASE_URL = saved_url
    
    main()
