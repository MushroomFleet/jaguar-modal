# Jaguar Diffusers with Modal Volumes

This document explains how the "jaguar-diffusers" text-to-image project has been improved by using Modal Volumes to store model weights.

## What are Modal Volumes?

Modal Volumes provide a high-performance distributed file system for Modal applications. They are designed for write-once, read-many I/O workloads, like creating machine learning model weights and distributing them for inference.

Benefits:
- **Faster startup**: Model weights are stored in the volume, eliminating the need to download them for each container
- **Reduced bandwidth**: Download models only once instead of on every container initialization
- **Better reliability**: Less dependency on external APIs (like HuggingFace) being available
- **Persistence**: Model weights remain available across deployments

## Implementation Details

Our implementation:
1. Creates a dedicated Modal Volume named `jaguar-model-weights`
2. Checks if the model exists in the volume during startup
3. If available, loads directly from the volume (fast)
4. If not available, downloads from HuggingFace and saves to the volume (one-time operation)
5. Provides utilities to force reload the model when needed

## Deploying the Application

To deploy the application with volume support:

```bash
# Deploy the application
modal deploy jaguar-modal/jaguar_app.py

# Note the deployment URL which will look like:
# https://yourname--shuttle-jaguar

# The first run will create the volume and download the model
```

## Testing the Implementation

Use the included test script to test the deployment:

```bash
# Set the API URL (only needed once)
cd jaguar-modal
python test-generate.py --url https://yourname--shuttle-jaguar

# Run subsequent tests (URL is saved)
python test-generate.py

# Display the generated image
python test-generate.py --display
```

The test script:
- Checks if the model is loading from the volume
- Generates an image with the prompt "white cat holding a sign reading Scuffed"
- Reports generation performance metrics

## Volume Management

You can manage the volume directly with the Modal CLI:

```bash
# List available volumes
modal volume list

# View details of the jaguar-model-weights volume
modal volume ls jaguar-model-weights

# To force a re-download of the model, use the API endpoint:
curl -X POST https://yourname--shuttle-jaguar-shuttlejaguarmodel-reload-model.modal.run
```

## Performance Benefits

With volume-based storage, you should see:
1. **First run**: Similar to original (~2-3 minutes) as it downloads and saves the model
2. **Subsequent runs**: Much faster startup (10-20 seconds) as it loads directly from volume

The app now also tracks and reports the model source (volume or HuggingFace) in the model info.

## Future Improvements

Possible future enhancements:
- Implement incremental model updates
- Add automatic version checking against HuggingFace
- Create separate volumes for different model versions
- Implement backup/restore functionality for volumes
