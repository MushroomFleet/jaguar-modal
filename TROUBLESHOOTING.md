# Troubleshooting Modal Deployment

This guide addresses common issues that may arise when deploying the Shuttle-Jaguar API using Modal.

## Common Deployment Issues

### PyTorch CUDA Dependencies Error

#### Problem 1: Invalid Index URL

**Error:**
```
WARNING: The index url "diffusers>=0.17.0" seems invalid, please provide a scheme.
ERROR: Cannot unpack file ... cannot detect archive format
```

**Solution:** This issue occurs with older versions of Modal when passing the `--extra-index-url` flag directly. The most reliable fix is to use `run_commands` for more direct control over the pip installation process:

```python
image = (
    modal.Image.debian_slim()
    .run_commands(
        "pip install torch==2.5.1+cu121 torchvision torchaudio diffusers>=0.17.0 transformers>=4.30.0 pillow>=9.0.0 'fastapi[standard]' --extra-index-url https://download.pytorch.org/whl/cu121"
    )
)
```

#### Problem 2: Invalid Requirement with extra_options

**Error:**
```
ERROR: Invalid requirement: '[--extra-index-url,': Expected package name at the start of dependency specifier
```

**Solution:** If you encounter this error while using the `extra_options` parameter, it indicates an issue with how Modal is passing the flags to pip. Instead of using `extra_options`, switch to the `run_commands` approach shown above.

### GPU Not Available

**Problem:** The model loads but GPU acceleration isn't working, or you get CUDA errors.

**Solution:** Make sure you're requesting a GPU in your Modal function or class:

```python
@app.cls(gpu="T4", image=image)  # Specifically request T4 GPU
```

### Model Loading Timeouts

**Problem:** Deployment times out during model downloading.

**Solution:** The Shuttle-Jaguar model is large. Consider:

1. Using a Modal volume to persist the model between runs:
```python
MODEL_VOLUME = modal.Volume.persisted("shuttle-jaguar-volume")

@app.cls(gpu="T4", image=image, volumes={"/models": MODEL_VOLUME})
class ShuttleJaguarModel:
    # ...
```

2. Set a longer timeout for the initial model download:
```bash
modal deploy --timeout 600 jaguar_app.py  # 10 minutes
```

### Memory Issues

**Problem:** Out of memory errors during image generation.

**Solution:** The model is using CPU offloading by default, but you can adjust batch sizes and image dimensions:

```python
# Reduce dimensions for larger batches
pipe(
    prompt,
    height=512,  # Smaller height
    width=512,   # Smaller width
    # ...
)
```

### Cold Start Latency

**Problem:** The first request after deployment is very slow.

**Solution:** This is expected as the model needs to be loaded. Options:

1. Use Modal's container idle timeout to keep the container warm:
```python
@app.cls(gpu="T4", image=image, container_idle_timeout=300)  # 5 minutes
```

2. Implement a warming endpoint that can be called periodically:
```python
@modal.method()
def warmup(self):
    # Load model if not already loaded
    if not hasattr(self, "pipe") or self.pipe is None:
        self.load_model()
    # Return simple status
    return {"status": "warmed up"}
```

## Debugging Tips

### Check Modal Logs

Modal provides detailed logs for your deployments:

```bash
modal app logs shuttle-jaguar
```

### Test Locally First

Before deploying, test your application locally:

```bash
modal serve --no-gpu jaguar_app.py  # Test without GPU
modal serve jaguar_app.py  # Test with GPU if available locally
```

### Inspect Container Environment

You can inspect the container environment using:

```python
@app.function()
def debug_environment():
    import os
    import sys
    import torch
    
    # Get environment info
    env_info = {
        "python_version": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "env_vars": dict(os.environ),
    }
    
    return env_info
```

### Check Resource Usage

Monitor GPU memory usage by adding logging:

```python
def check_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
        }
    return {"error": "CUDA not available"}
```

## Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/index)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/notes/cuda.html)
