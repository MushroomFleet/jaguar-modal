# Shuttle-Jaguar Modal API

A serverless API for the shuttle-jaguar text-to-image model using Modal.

## Overview

This implementation uses Modal to host the shuttle-jaguar text-to-image model as a serverless API, enabling image generation through HTTP requests. The model uses the FP8 format for optimal performance and runs on T4 GPUs.

## Features

- **Automatic Scaling**: Modal handles scaling based on demand
- **GPU Acceleration**: Uses T4 GPUs for fast inference
- **Multiple API Endpoints**:
  - Single image generation
  - Batch processing for multiple prompts
  - Model information
- **Base64 Image Encoding**: Generated images are returned as base64-encoded data
- **Parameter Customization**: Control height, width, guidance scale, steps, etc.

## Installation

1. Install the Modal CLI:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

3. Clone this repository:
```bash
git clone https://github.com/MushroomFleet/jaguar-modal
cd jaguar-modal
```

## Usage

### Deploy the API

```bash
modal deploy jaguar_app.py
```

### Development and Testing

Run the server locally during development:

```bash
modal serve jaguar_app.py
```

### API Endpoints

#### 1. Generate a single image

```
GET /ShuttleJaguarModel/generate_api?prompt=your_prompt_here
```

Optional parameters:
- `height`: Image height (default: 1024)
- `width`: Image width (default: 1024)
- `guidance_scale`: Classifier-free guidance scale (default: 3.5)
- `steps`: Number of inference steps (default: 4)
- `max_seq_length`: Maximum sequence length (default: 256)
- `seed`: Random seed for reproducibility

#### 2. Batch generate multiple images

```
POST /ShuttleJaguarModel/batch_api
```

Request body (JSON):
```json
{
  "prompts": ["prompt1", "prompt2", "prompt3"],
  "height": 1024,
  "width": 1024,
  "guidance_scale": 3.5,
  "steps": 4,
  "max_seq_length": 256,
  "base_seed": 42
}
```

#### 3. Get model information

```
GET /ShuttleJaguarModel/info
```

### Using the API from Python

```python
import modal
import base64
from PIL import Image
import io

# Deploy the app first with `modal deploy jaguar_app.py`
app = modal.App("shuttle-jaguar")
model = app.ShuttleJaguarModel()

# Generate an image
result = model.generate_image.remote(
    prompt="A beautiful mountain landscape",
    width=768,
    height=768
)

# Convert base64 image to a PIL Image
image_data = base64.b64decode(result["image"])
image = Image.open(io.BytesIO(image_data))
image.save("output.png")
```

### Using the API with curl

```bash
# Generate a single image
curl -X GET "https://your-modal-deployment--shuttlejaguarmodel-generate-api.modal.run?prompt=A%20beautiful%20mountain%20landscape"

# Batch generate multiple images
curl -X POST "https://your-modal-deployment--shuttlejaguarmodel-batch-api.modal.run" \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["A cat", "A dog", "A bird"]}'
```

## Model Information

- **Model**: shuttle-jaguar (FP8 format)
- **Provider**: shuttleai
- **Parameters**: 8B
- **Default Settings**:
  - Height: 1024px
  - Width: 1024px
  - Guidance Scale: 3.5
  - Steps: 4
  - Max Sequence Length: 256

## Technical Details

- The model is loaded once during container initialization using `modal.enter()`
- CPU offloading is enabled to optimize VRAM usage
- All API responses include timing information
- Images are returned as base64-encoded strings for easy embedding in web pages

## Requirements

- Python 3.7+
- Modal account
- Internet connection (for downloading the model on first run)
