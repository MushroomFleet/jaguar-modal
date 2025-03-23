import base64
import io
import modal
from typing import Optional, List, Union
import torch

# Create the Modal app
app = modal.App("shuttle-jaguar")

# Define the image with required dependencies
image = (
    modal.Image.debian_slim()
    .run_commands(
        "pip install torch==2.5.1+cu121 torchvision torchaudio diffusers>=0.17.0 transformers>=4.30.0 pillow>=9.0.0 'fastapi[standard]' accelerate --extra-index-url https://download.pytorch.org/whl/cu121"
    )
)

@app.cls(gpu="T4", image=image)
class ShuttleJaguarModel:
    """Shuttle-Jaguar model for text-to-image generation served via Modal."""
    
    @modal.enter()
    def load_model(self):
        """Load the model during container initialization."""
        from diffusers import DiffusionPipeline
        import torch

        print("Loading Shuttle-Jaguar model...")
        
        try:
            # Load the model with appropriate settings for efficiency
            # Removed variant="fp8" as it's not available for this model
            self.pipe = DiffusionPipeline.from_pretrained(
                "shuttleai/shuttle-jaguar", 
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True
            ).to("cuda")
            
            # Enable CPU offload for VRAM optimization
            self.pipe.enable_model_cpu_offload()
            
            print("Model loaded successfully!")
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
        return {
            "model": "shuttleai/shuttle-jaguar",
            "version": "bfloat16",
            "parameters": "8B",
            "format": "diffusers",
            "capabilities": [
                "text-to-image"
            ],
            "recommended_settings": {
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
                "num_steps": 4,
                "max_seq_length": 256
            }
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


# Example of how to call the model functions directly from Python
@app.function(image=image)
def main():
    model = ShuttleJaguarModel()
    
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


if __name__ == "__main__":
    # For local development and testing
    # When running modal serve jaguar_app.py, the web endpoints will be available
    # For production, use modal deploy jaguar_app.py
    main.local()
