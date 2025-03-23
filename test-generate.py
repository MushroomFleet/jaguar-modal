#!/usr/bin/env python
"""
Test script for the Shuttle-Jaguar Modal API with volume support.
This script runs a test image generation with a specific prompt.
"""

import base64
import io
import os
import requests
import json
from PIL import Image
import time
import argparse

# The URL needs to be updated with your actual deployment URL
BASE_URL = "https://mushroomfleet--shuttle-jaguar"

def test_generate(display=False):
    """Run a test generation with a specific prompt."""
    # Test parameters
    prompt = "anthropmorphic frog holding a sign reading Epoch, on a lily pad"
    output_file = "test_output.png"
    
    # API parameters
    params = {
        "prompt": prompt,
        "height": 1024,
        "width": 1024,
        "guidance_scale": 4.5,
        "steps": 12,
        "max_seq_length": 256
    }
    
    # First, get model info to check if using volume
    print("Checking model information...")
    info_url = f"{BASE_URL}-shuttlejaguarmodel-info.modal.run"
    
    try:
        info_response = requests.get(info_url, timeout=30)
        if info_response.status_code == 200:
            model_info = info_response.json()
            print("\nModel Information:")
            print(f"- Name: {model_info.get('model', 'unknown')}")
            print(f"- Parameters: {model_info.get('parameters', 'unknown')}")
            print(f"- Source: {model_info.get('source', 'unknown')}")
            if 'source' in model_info and model_info['source'] == 'volume':
                print("✅ Model is loading from volume!")
            else:
                print("⚠️ Model is not loading from volume.")
            print(f"- Volume path: {model_info.get('volume_path', 'unknown')}")
        else:
            print(f"❌ Error getting model info: {info_response.status_code}")
    except Exception as e:
        print(f"❌ Error connecting to info endpoint: {e}")
    
    # Make the generation request
    url = f"{BASE_URL}-shuttlejaguarmodel-generate-api.modal.run"
    
    print(f"\nGenerating image with prompt: '{prompt}'")
    start_time = time.time()
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
        
        result = response.json()
        generation_time = time.time() - start_time
        
        # Save the image
        if "image" in result:
            image_data = base64.b64decode(result["image"])
            image = Image.open(io.BytesIO(image_data))
            image.save(output_file)
            print(f"✅ Image saved to {output_file}")
            
            # Get the reported generation time from the API
            api_gen_time = result.get("generation_time", "unknown")
            print(f"\nPerformance:")
            print(f"- API reported generation time: {api_gen_time} seconds")
            print(f"- Total round-trip time: {generation_time:.2f} seconds")
            
            # Display the image if requested
            if display:
                image.show()
                
            return True
        else:
            print("❌ Error: No image in response")
            print(json.dumps(result, indent=2))
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False

def update_url(url):
    """Update the base URL for the API."""
    # Save the URL to a config file for persistence
    with open(os.path.join(os.path.dirname(__file__), '.api_url'), 'w') as f:
        f.write(url)
    print(f"API URL set to: {url}")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the Shuttle-Jaguar Modal API")
    parser.add_argument("--url", type=str, help="Set the base URL for the API")
    parser.add_argument("--display", action="store_true", help="Display the generated image")
    args = parser.parse_args()
    
    # Try to load saved URL if it exists
    config_path = os.path.join(os.path.dirname(__file__), '.api_url')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_url = f.read().strip()
            if saved_url:
                BASE_URL = saved_url
                
    # Update URL if provided
    if args.url:
        update_url(args.url)
        BASE_URL = args.url
        
    # Check if we have a valid URL
    if BASE_URL == "https://yourname--shuttle-jaguar":
        print("⚠️ Warning: You need to set your actual Modal deployment URL!")
        print("Run: python test-generate.py --url YOUR_DEPLOYMENT_URL")
        print("Your URL will be saved for future use.")
        exit(1)
        
    # Run the test
    test_generate(display=args.display)
