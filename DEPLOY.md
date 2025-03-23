# Deploying the Shuttle-Jaguar Modal API

This guide walks you through deploying the Shuttle-Jaguar image generation API to Modal, addressing common issues, and verifying the installation.

## Prerequisites

1. Install Modal and authenticate
   ```bash
   pip install modal
   modal token new
   ```

2. Ensure you have a Modal account with GPU access

## Deployment Steps

1. Deploy the application
   ```bash
   cd jaguar-modal
   modal deploy jaguar_app.py
   ```

2. Modal will build the image and deploy the application, providing URLs for the endpoints:
   ```
   ✓ Created objects.
   ├── 🔨 Created mount [path]
   ├── 🔨 Created function ShuttleJaguarModel.*.
   ├── 🔨 Created function main.
   ├── 🔨 Created web endpoint for ShuttleJaguarModel.generate_api
   ├── 🔨 Created web endpoint for ShuttleJaguarModel.info
   └── 🔨 Created web endpoint for ShuttleJaguarModel.batch_api
   ✓ App deployed! 🎉
   ```

3. Update the client with the correct endpoint URLs:
   ```bash
   cd jaguar-modal
   python fix_client.py your-deployment-url
   ```

## Testing the Deployment

1. Test the info endpoint:
   ```bash
   python client_example.py info
   ```

2. Generate a simple image:
   ```bash
   python client_example.py generate "A simple test image" --width 512 --height 512
   ```

3. For batch generation:
   ```bash
   python client_example.py batch --prompts "Prompt 1" "Prompt 2" "Prompt 3"
   ```

## Recent Fixes

### Fixed Dependencies and Model Loading

The current implementation includes these important fixes:

1. **Added Accelerate Library**: 
   - The `accelerate` library is now included in dependencies
   - Provides faster and memory-efficient model loading
   - Enables `low_cpu_mem_usage=True` parameter

2. **Corrected Model Parameters**:
   - Removed unsupported `variant="fp8"` parameter
   - Using `torch_dtype=torch.bfloat16` for efficient processing
   - Added `use_safetensors=True` for better memory usage

3. **Added Error Handling**:
   - Robust try/except blocks for model loading
   - Better diagnostics for deployment issues

## Configuration Options

### Memory Optimization

The model uses CPU offloading to optimize VRAM usage:

```python
self.pipe.enable_model_cpu_offload()
```

This keeps some model components in CPU memory and transfers them to GPU only when needed.

### Performance Tuning

You can adjust these parameters for different performance characteristics:

- `guidance_scale`: Controls creativity vs. adherence to prompt (default: 3.5)
- `num_inference_steps`: More steps = higher quality but slower (default: 4)
- `height`/`width`: Smaller dimensions use less memory (defaults: 1024x1024)

## Monitoring and Logs

To view the logs for your deployment:

```bash
modal app logs shuttle-jaguar
```

This shows all server-side logs, including model loading, inference times, and errors.
