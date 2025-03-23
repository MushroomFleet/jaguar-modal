import base64
import io
import os
import modal
from typing import Optional, List, Union
import torch

# Create the Modal app
app = modal.App("shuttle-jaguar")

# Define a volume to store model weights
model_volume = modal.Volume.from_name("jaguar-model-weights", create_if_missing=True)
MODEL_MOUNT_PATH = "/vol/models/shuttle-jaguar"

# Define the image with required dependencies
image = (
    modal.Image.debian_slim()
    .run_commands(
        "pip install torch==2.5.1+cu121 torchvision torchaudio diffusers>=0.17.0 transformers>=4.30.0 pillow>=9.0.0 'fastapi[standard]' accelerate sentencepiece --extra-index-url https://download.pytorch.org/whl/cu121"
    )
)

@app.cls(gpu="A100-40GB", image=image, volumes={MODEL_MOUNT_PATH: model_volume})
class ShuttleJaguarModel:
    """Shuttle-Jaguar model for text-to-image generation served via Modal."""
    
    def _model_exists_in_volume(self):
        """Check if the model exists in the volume."""
        model_dir = MODEL_MOUNT_PATH
        # Check for the main config file which should exist in a valid model
        config_file = os.path.join(model_dir, "model_index.json")
        return os.path.exists(config_file)
    
    @modal.enter()
    def load_model(self):
        """Load the model during container initialization."""
        from diffusers import DiffusionPipeline
        import torch
        import time

        start_time = time.time()
        
        # Model source and destination paths
        model_id = "shuttleai/shuttle-jaguar"
        local_model_path = MODEL_MOUNT_PATH

        print("Loading Shuttle-Jaguar model...")
        
        try:
            if self._model_exists_in_volume():
                print(f"Loading model from volume at: {local_model_path}")
                # Load model from volume
                self.pipe = DiffusionPipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                ).to("cuda")
                
                print(f"Model loaded from volume in {time.time() - start_time:.2f} seconds")
            else:
                print(f"Model not found in volume. Downloading from HuggingFace: {model_id}")
                # Download model from HuggingFace
                self.pipe = DiffusionPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                ).to("cuda")
                
                print(f"Saving model to volume at: {local_model_path}")
                # Save model to volume for future use
                self.pipe.save_pretrained(local_model_path)
                
                # Commit changes to volume
                model_volume.commit()
                print(f"Model saved to volume successfully in {time.time() - start_time:.2f} seconds")
            
            # Enable CPU offload for VRAM optimization in either case
            self.pipe.enable_model_cpu_offload()
            
            print(f"Model fully initialized and ready in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Re-raise to ensure Modal knows there was a problem
            raise
    
    def _generate_image(
        self, 
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_steps: int = 4,
        max_seq_length: int = 256,
        seed: Optional[int] = None,
    ):
        """Internal method to generate an image."""
        # Set up seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        
        # Generate the image
        return self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            max_sequence_length=max_seq_length,
            generator=generator
        ).images[0]
    
    def _encode_image(self, image):
        """Convert PIL image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    @modal.method()
    def generate_image(
        self, 
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_steps: int = 4,
        max_seq_length: int = 256,
        seed: Optional[int] = None,
        return_base64: bool = True
    ):
        """Generate an image and return it as base64 or save it to file."""
        import time
        
        # Track time for generation
        start_time = time.time()
        
        # Generate the image
        image = self._generate_image(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            max_seq_length=max_seq_length,
            seed=seed
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Return as base64 if requested
        if return_base64:
            return {
                "image": self._encode_image(image),
                "parameters": {
                    "prompt": prompt,
                    "height": height,
                    "width": width,
                    "guidance_scale": guidance_scale,
                    "num_steps": num_steps,
                    "max_seq_length": max_seq_length,
                    "seed": seed
                },
                "generation_time": round(generation_time, 2)
            }
        else:
            # Return the PIL image
            return image
    
    @modal.method()
    def batch_generate(
        self,
        prompts: List[str],
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_steps: int = 4,
        max_seq_length: int = 256,
        base_seed: Optional[int] = None
    ):
        """Generate multiple images from a list of prompts."""
        import time
        
        results = []
        total_start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            # Set seed for this prompt (increment base_seed if provided)
            seed = None
            if base_seed is not None:
                seed = base_seed + i
            
            # Generate image for this prompt
            start_time = time.time()
            image = self._generate_image(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                max_seq_length=max_seq_length,
                seed=seed
            )
            generation_time = time.time() - start_time
            
            # Add to results
            results.append({
                "prompt": prompt,
                "image": self._encode_image(image),
                "seed": seed,
                "generation_time": round(generation_time, 2)
            })
        
        # Calculate total generation time
        total_time = time.time() - total_start_time
        
        return {
            "results": results,
            "parameters": {
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "num_steps": num_steps,
                "max_seq_length": max_seq_length,
                "base_seed": base_seed
            },
            "total_generation_time": round(total_time, 2),
            "images_generated": len(prompts)
        }
    
    @modal.method()
    def get_model_info(self):
        """Return information about the model."""
        # Check if model is loaded from volume
        model_source = "volume" if self._model_exists_in_volume() else "huggingface"
        
        return {
            "model": "shuttleai/shuttle-jaguar",
            "version": "bfloat16",
            "parameters": "8B",
            "format": "diffusers",
            "source": model_source,
            "capabilities": [
                "text-to-image"
            ],
            "recommended_settings": {
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
                "num_steps": 4,
                "max_seq_length": 256
            },
            "volume_path": MODEL_MOUNT_PATH
        }
    
    @modal.method()
    def force_model_reload(self):
        """Force reload the model from HuggingFace and save to volume.
        
        This can be useful to update the model to a newer version.
        """
        from diffusers import DiffusionPipeline
        import torch
        import time
        
        start_time = time.time()
        model_id = "shuttleai/shuttle-jaguar"
        local_model_path = MODEL_MOUNT_PATH
        
        print(f"Force downloading model from HuggingFace: {model_id}")
        try:
            # Download model from HuggingFace
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                force_download=True
            ).to("cuda")
            
            print(f"Re-saving model to volume at: {local_model_path}")
            # Save model to volume
            self.pipe.save_pretrained(local_model_path)
            
            # Commit changes to volume
            model_volume.commit()
            
            # Enable CPU offload for VRAM optimization
            self.pipe.enable_model_cpu_offload()
            
            total_time = time.time() - start_time
            print(f"Model reloaded and saved to volume in {total_time:.2f} seconds")
            
            return {
                "success": True,
                "message": f"Model successfully reloaded in {total_time:.2f} seconds",
                "model_path": local_model_path
            }
        except Exception as e:
            error_message = f"Error reloading model: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "message": error_message
            }
    
    # FastAPI endpoints
    @modal.fastapi_endpoint()
    def generate_api(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        steps: int = 4,
        max_seq_length: int = 256,
        seed: Optional[int] = None
    ):
        """API endpoint for generating a single image."""
        result = self.generate_image.remote(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_steps=steps,
            max_seq_length=max_seq_length,
            seed=seed,
            return_base64=True
        )
        return result
    
    @modal.fastapi_endpoint(method="POST")
    def batch_api(self, data: dict):
        """API endpoint for batch generating multiple images."""
        # Extract parameters from request body
        prompts = data.get("prompts", [])
        if not prompts:
            return {"error": "No prompts provided"}
        
        # Get other parameters with defaults
        height = data.get("height", 1024)
        width = data.get("width", 1024)
        guidance_scale = data.get("guidance_scale", 3.5)
        steps = data.get("steps", 4)
        max_seq_length = data.get("max_seq_length", 256)
        base_seed = data.get("base_seed")
        
        # Call the batch generation method
        return self.batch_generate.remote(
            prompts=prompts,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_steps=steps,
            max_seq_length=max_seq_length,
            base_seed=base_seed
        )
    
    @modal.fastapi_endpoint()
    def info(self):
        """API endpoint for getting model information."""
        return self.get_model_info.remote()
    
    @modal.fastapi_endpoint(method="POST")
    def reload_model(self):
        """API endpoint to force reload the model from HuggingFace."""
        return self.force_model_reload.remote()


# Example of how to call the model functions directly from Python
@app.function(image=image)
def main():
    model = ShuttleJaguarModel()
    
    # First, get model information to check the source (volume or huggingface)
    info = model.get_model_info.remote()
    print(f"Model information:")
    print(f"  - Name: {info['model']}")
    print(f"  - Parameters: {info['parameters']}")
    print(f"  - Source: {info['source']}")
    print(f"  - Volume path: {info['volume_path']}")
    print()
    
    # Example: Generate a single image
    result = model.generate_image.remote(
        prompt="A cat holding a sign that says hello world",
        width=768,
        height=768
    )
    
    print(f"Generated image with parameters: {result['parameters']}")
    print(f"Generation time: {result['generation_time']} seconds")
    
    # The result contains a base64 encoded image that can be decoded
    # or used directly in HTML with: <img src="data:image/png;base64,{result['image']}">
    
    print("\nNote: The first run will download the model from HuggingFace")
    print("and save it to the volume. Subsequent runs will be faster")
    print("as the model will be loaded directly from the volume.")
    print("\nTo force a model update, use the reload_model endpoint:")
    print("  POST /reload_model")


# Utility function to create or initialize the volume
@app.function(image=modal.Image.debian_slim())
def create_volume():
    """Create or ensure the model volume exists.
    
    This function can be run separately to set up the volume
    before running the main application.
    """
    # Ensure volume exists
    vol = modal.Volume.from_name("jaguar-model-weights", create_if_missing=True)
    
    # Check if it exists
    vol_info = vol.get()
    
    # Get volume size
    size_mb = vol_info.get("size_mb", 0)
    
    print(f"Volume 'jaguar-model-weights' is ready.")
    print(f"Current size: {size_mb:.2f} MB")
    print(f"Mount path in containers: {MODEL_MOUNT_PATH}")
    
    if size_mb < 1:
        print("Volume is empty. When you run the app, the model will be")
        print("downloaded from HuggingFace and saved to the volume.")
    else:
        print("Volume contains data. The app will attempt to load the model from the volume.")
    
    return {
        "name": "jaguar-model-weights",
        "size_mb": size_mb,
        "mount_path": MODEL_MOUNT_PATH
    }


if __name__ == "__main__":
    # For local development and testing
    # When running modal serve jaguar_app.py, the web endpoints will be available
    # For production, use modal deploy jaguar_app.py
    
    # Uncomment to create/check the volume first:
    # create_volume.local()
    
    main.local()
