# VIBE Local GUI (Windows) 🎨

A local Gradio Web-UI for the [VIBE model](https://huggingface.co/iitolstykh/VIBE-Image-Edit), offering advanced controls like Seed, Sample Steps, and Guidance Scale adjustments.

**Tested on Windows 11 with NVIDIA RTX 5090 (Blackwell architecture).**

![Preview](preview.png)
*(Run your own local instance with full privacy and speed)*

## ✨ Features
- 🚀 **Runs Locally:** fast, private, and no limits.
- 🎛️ **Full Control:** Adjust `Sample Steps`, `Seed`, `Guidance Scale`, and `Image Guidance`.
- 💾 **Auto-Save:** Results are automatically saved to the `outputs` folder with timestamps.
- 🛠️ **Windows Optimized:** Fixes common pathing and dependency issues found in the official repo.

## ⚙️ Installation

### 1. Prerequisites
- **Python 3.10** or **3.11** installed (Make sure to check "Add Python to PATH" during installation).
- **Git** installed.
- An NVIDIA GPU (16GB+ VRAM recommended, 24GB+ for best performance).

### 2. Clone the Repository
Open PowerShell or Terminal and run:

```bash
git clone https://github.com/Detoxfox4234/VIBE-Local-GUI-Windows.git
cd VIBE-Local-GUI-Windows
```

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment to avoid conflicts.

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
This will install `gradio`, `torch`, and the VIBE model requirements.

```bash
pip install -r requirements.txt
```

⚠️ **Important for RTX 5090 Users (Blackwell)**

The standard PyTorch version installed by `requirements.txt` might not yet support the RTX 50 series (Compute Capability sm_120).

If you encounter a `Torch not compiled with CUDA enabled` or `sm_120` error, please reinstall PyTorch with the Nightly Build manually:

```bash
# 1. Uninstall the standard version
pip uninstall torch torchvision torchaudio -y

# 2. Install the Nightly version (supports CUDA 12.8+)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 🚀 Usage

1. Make sure your environment is active (`(venv)` should be visible).
2. Run the application:

```bash
python app.py
```

3. Wait for the model to load. Your default web browser will open automatically at `http://127.0.0.1:7860`.

**Note:** On the very first run, the script will download the VIBE model weights (~10GB) from Hugging Face. This might take a while depending on your internet connection.

## 📂 Output

All generated images are saved in the `outputs` folder inside the installation directory. The filenames include the timestamp and the seed used, so you can easily reproduce results.

## 🔗 Credits

* Original VIBE Model: ai-forever/VIBE
* Model Weights: Hugging Face

## 🤝 Support

This is a free open-source project. I don't ask for donations.
However, if you want to say "Thanks", check out my profile on **Spotify**.
A follow or a listen is the best way to support me! 🎧

👉 [Listen to my Music on Spotify](https://open.spotify.com/artist/7EdK2cuIo7xTAacutHs9gv?si=C99w8i5jRBKwcjutpIWInQ)
