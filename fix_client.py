#!/usr/bin/env python
"""
Script to fix the client configuration for the Shuttle-Jaguar Modal API.
This will update the client_example.py file with the correct endpoint URLs.
"""

import os
import re
import sys

def fix_client_config(base_url=None):
    """Fix the client configuration with correct endpoint URLs."""
    client_file = "client_example.py"
    
    if not os.path.exists(client_file):
        print(f"Error: {client_file} not found in the current directory.")
        return False
    
    # Read the current client file
    with open(client_file, 'r') as f:
        content = f.read()
    
    # If no base_url provided, use the one from deployment output
    if not base_url:
        base_url = "https://mushroomfleet--shuttle-jaguar"
    
    # Endpoints as shown in the deployment message
    info_endpoint = f"{base_url}-shuttlejaguarmodel-info.modal.run"
    generate_endpoint = f"{base_url}-shuttlejaguarmodel-generate-api.modal.run"
    batch_endpoint = f"{base_url}-shuttlejaguarmodel-batch-api.modal.run"
    
    # Create updated content with hard-coded endpoints
    updated_content = re.sub(r'BASE_URL = ".*?"', f'BASE_URL = "{base_url}"', content)
    
    # Replace the URL construction with direct endpoint URLs
    updated_content = re.sub(
        r'url = f"{BASE_URL}-shuttlejaguarmodel-generate-api\.modal\.run"',
        f'url = "{generate_endpoint}"',
        updated_content
    )
    
    updated_content = re.sub(
        r'url = f"{BASE_URL}-shuttlejaguarmodel-batch-api\.modal\.run"',
        f'url = "{batch_endpoint}"',
        updated_content
    )
    
    updated_content = re.sub(
        r'url = f"{BASE_URL}-shuttlejaguarmodel-info\.modal\.run"',
        f'url = "{info_endpoint}"',
        updated_content
    )
    
    # Add a comment explaining the direct endpoint URLs
    updated_content = updated_content.replace(
        "# Replace with your actual deployment URL after deploying",
        "# Deployment URLs for the shuttle-jaguar model endpoints"
    )
    
    # Save the updated file
    with open(client_file, 'w') as f:
        f.write(updated_content)
    
    # Create a .api_url file with the base URL
    with open('.api_url', 'w') as f:
        f.write(base_url)
    
    print(f"Client configuration updated successfully!")
    print(f"Base URL: {base_url}")
    print(f"Info endpoint: {info_endpoint}")
    print(f"Generate endpoint: {generate_endpoint}")
    print(f"Batch endpoint: {batch_endpoint}")
    return True

if __name__ == "__main__":
    # If a command-line argument is provided, use it as the base URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else None
    fix_client_config(base_url)
