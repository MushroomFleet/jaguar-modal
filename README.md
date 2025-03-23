# 🚀✨ SHUTTLE JAGUAR DIFFUSERS API ✨🚀

> *Your gateway to AI-powered imagination mayhem!* 🎭🎨

Generate mind-bending images with the power of the Shuttle-Jaguar text-to-image model in a **SERVERLESS** ☁️ environment using Modal! No more GPU nightmares! No more dependency hell! Just pure, unadulterated image generation MADNESS! 🤯

![Shuttle-Jaguar](https://img.shields.io/badge/Shuttle--Jaguar-8B%20Parameters-blueviolet) ![GPU](https://img.shields.io/badge/GPU-A100--40GB-brightgreen) ![Modal](https://img.shields.io/badge/Platform-Modal-blue)

## 🤔 What is This Madness? 🤔

This project lets you deploy the **Shuttle-Jaguar model** (that's a whopping 8 BILLION parameters, folks! 🤩) as a serverless API using Modal. It handles all the infrastructure so you can focus on generating images that will make your eyeballs tingle with joy! ✨👁️✨

The latest version uses **Modal Volumes** to store model weights, making everything FASTER 🏎️💨 and MORE RELIABLE 🔒 than ever before!

## ✨ Features That'll Blow Your Mind ✨

- 🔥 **SERVERLESS DEPLOYMENT**: No servers to manage! Modal handles EVERYTHING!
- 🧠 **BIG BRAIN MODEL**: 8B parameter Shuttle-Jaguar model for INCREDIBLE image generation!
- 🚄 **LIGHTNING FAST**: Uses A100-40GB GPUs for ridiculously fast inference!
- 📦 **VOLUME STORAGE**: Store model weights for faster startup times!
- 🔄 **AUTO-SCALING**: Handles as many requests as you throw at it!
- 🔌 **MULTIPLE ENDPOINTS**: Single image, batch processing, model info - WE GOT IT ALL!
- 🎛️ **CUSTOMIZATION**: Control height, width, guidance - TWEAK ALL THE THINGS!
- 🔋 **OPTIMIZED**: CPU offloading, bfloat16 precision, low memory usage!

## 🛠️ Prerequisites Before The Magic Happens 🛠️

- 💻 Python 3.7+ (the fresher the better!)
- 🔑 Modal account (sign up at [modal.com](https://modal.com) - it's FREE to start!)
- 🌐 Internet connection (to download the model on first run)
- 🤩 An imagination ready to be UNLEASHED!

## 🚀 Installation: Let's Summon The Beast 🚀

1. 📦 **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. 🔐 **Authenticate with Modal**:
   ```bash
   modal token new
   ```

3. 📥 **Clone this repository**:
   ```bash
   git clone https://github.com/MushroomFleet/jaguar-diffusers
   cd jaguar-diffusers
   ```

## 🧙‍♂️ Deployment: The Grand Conjuring 🧙‍♂️

1. 🚀 **Deploy the Application**:
   ```bash
   modal deploy jaguar_app.py
   ```

2. 📝 **Note Your Deployment URL**:
   ```
   https://yourname--shuttle-jaguar
   ```

3. 🧪 **Test the Deployment**:
   ```bash
   cd jaguar-modal
   python test-generate.py --url https://yourname--shuttle-jaguar
   ```

4. 🎉 **MARVEL at your Creation!**:
   ```bash
   python test-generate.py --display
   ```

> 🔍 **FIRST RUN WARNING**: The first time you run, the model will download from HuggingFace and save to a volume (~2-3 minutes). Subsequent runs will be MUCH faster (~10-20 seconds). PATIENCE, YOUNG WIZARD! ⏳

For more deployment details, check [jaguar-modal/DEPLOY.md](jaguar-modal/DEPLOY.md).

## 🔮 Usage: Unleash The Image-Creating Kraken 🔮

### 🖼️ Generate a Single Image

```bash
# Using the test script (EASIEST WAY)
python test-generate.py

# Using curl (for the COMMAND LINE WIZARDS)
curl -X GET "https://yourname--shuttle-jaguar-shuttlejaguarmodel-generate-api.modal.run?prompt=A%20magical%20forest%20with%20glowing%20mushrooms"

# Using the Python client (for the CODE SORCERERS)
python client_example.py generate "A magical forest with glowing mushrooms" --width 768 --height 768
```

### 📚 Batch Generate Multiple Images

```bash
# Using the client
python client_example.py batch --prompts "A cat wizard" "A dog astronaut" "A rabbit pirate"
```

### 🔍 Check Model Information

```bash
python client_example.py info
```

### 🧨 Force Model Reload (DANGEROUS POWER!)

```bash
curl -X POST "https://yourname--shuttle-jaguar-shuttlejaguarmodel-reload-model.modal.run"
```

## 🌟 API Endpoints: For the Tech-Savvy Magicians 🌟

### 1. 🖼️ Generate Single Image
```
GET /ShuttleJaguarModel/generate_api?prompt=your_prompt_here
```
Parameters:
- `prompt`: Your imagination in words ✨
- `height`: Image height (default: 1024)
- `width`: Image width (default: 1024)
- `guidance_scale`: Creativity control (default: 3.5)
- `steps`: Quality control (default: 4)
- `max_seq_length`: Text length limit (default: 256)
- `seed`: Reproducibility magic (optional)

### 2. 📚 Batch Generate Images
```
POST /ShuttleJaguarModel/batch_api
```
JSON body:
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

### 3. ℹ️ Get Model Information
```
GET /ShuttleJaguarModel/info
```

### 4. 🔄 Force Model Reload
```
POST /ShuttleJaguarModel/reload_model
```

## 📊 Performance: It's FAST, Like REALLY FAST 📊

With the Modal Volumes implementation, you get:

- 🚀 **FASTER STARTUP**: No more downloading the model every time!
- 🌐 **REDUCED BANDWIDTH**: Download once, use forever!
- 🔒 **BETTER RELIABILITY**: Less dependency on external APIs!
- 🧠 **SMART LOADING**: Automatically uses volume if available!

```
First run: ~2-3 minutes (downloads & saves model)
Subsequent runs: ~10-20 seconds (loads from volume)
```

For volume implementation details, check [jaguar-modal/VOLUME.md](jaguar-modal/VOLUME.md).

## 🩺 Troubleshooting: When The Magic Goes Sideways 🩺

### 💥 Memory Issues

If you're getting OOM (Out Of Memory) errors:
- 📏 Reduce image dimensions (try 512x512)
- 🔢 Use fewer steps (try 2-3)
- 📉 Lower batch size

### 🐌 Slow First Run

- 😴 This is normal! The model is being downloaded and saved to the volume.
- ⏱️ Subsequent runs will be MUCH faster!

### 🔥 Deployment Failed

- 🔍 Check your CUDA dependencies
- 🧹 Make sure you have quota/credits on Modal
- 📋 Check logs with `modal app logs shuttle-jaguar`

For more troubleshooting wisdom, consult [jaguar-modal/TROUBLESHOOTING.md](jaguar-modal/TROUBLESHOOTING.md).

## 📚 Documentation: The Sacred Texts 📚

- [jaguar-modal/DEPLOY.md](jaguar-modal/DEPLOY.md): Deployment steps and configuration
- [jaguar-modal/TROUBLESHOOTING.md](jaguar-modal/TROUBLESHOOTING.md): Fixing common issues
- [jaguar-modal/VOLUME.md](jaguar-modal/VOLUME.md): Modal Volumes implementation
- Code files:
  - [jaguar-modal/jaguar_app.py](jaguar-modal/jaguar_app.py): Main application code
  - [jaguar-modal/client_example.py](jaguar-modal/client_example.py): Python client example
  - [jaguar-modal/test-generate.py](jaguar-modal/test-generate.py): Simple test script

## 📊 Model Information 📊

- **Model**: 🧠 shuttle-jaguar
- **Format**: 🧮 bfloat16
- **Parameters**: 🔢 8B (that's BILLION!)
- **GPU Used**: 🔥 A100-40GB
- **Default Settings**:
  - Height: 1024px
  - Width: 1024px
  - Guidance Scale: 3.5
  - Steps: 4
  - Max Sequence Length: 256

## 🙏 Acknowledgments 🙏

- 🌟 [ShuttleAI](https://huggingface.co/shuttleai) for the amazing Shuttle-Jaguar model
- 🚀 [Modal](https://modal.com) for the incredible serverless platform
- 🤗 [Hugging Face](https://huggingface.co) for hosting diffusion models
- 🍄 The MushroomFleet community for support and madness!

---

> 💫 Made with ✨COSMIC ENERGY✨ and probably too much caffeine ☕☕☕
