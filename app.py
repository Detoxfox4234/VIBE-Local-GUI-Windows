import gradio as gr
import torch
from PIL import Image
from vibe.editor import ImageEditor
import os
import time
import random

# --- CONFIGURATION ---
# Using relative paths for cross-platform compatibility
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# Looking for local model folder or fallback to HuggingFace
MODEL_PATH = "VIBE_Model" 

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
print("Starting GUI...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device}")

# Checkpoint path logic:
# Use local "VIBE_Model" if exists, otherwise download from HuggingFace
checkpoint = MODEL_PATH if os.path.exists(MODEL_PATH) else "iitolstykh/VIBE-Image-Edit"

print(f"Loading model from: {checkpoint}")
try:
    editor = ImageEditor(checkpoint_path=checkpoint)
    # Move the pipeline to the detected device
    _ = editor.pipe.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have installed: pip install git+https://github.com/ai-forever/VIBE")
    exit()

print("Ready! Launching interface...")

# --- CORE FUNCTION ---
def process_image(input_image, instruction, steps, guidance_scale, image_guidance_scale, seed):
    if input_image is None:
        return None, "No image uploaded"
    
    # Convert numpy array to PIL Image
    image = Image.fromarray(input_image).convert("RGB")
    
    # Random seed handling
    if seed == -1:
        seed = random.randint(0, 2147483647)
    
    print(f"Generating: '{instruction}' | Seed: {seed} | Steps: {steps}")

    # Initialize generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    try:
        # Calling the pipeline directly for advanced parameter support
        edited_image = editor.pipe(
            prompt=instruction,
            conditioning_image=image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            image_guidance_scale=float(image_guidance_scale),
            generator=generator
        ).images[0]
        
        # Save logic
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"VIBE_{timestamp}_seed{seed}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        edited_image.save(save_path, quality=100)
        return edited_image, f"Saved to: {save_path}"
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return None, f"Error: {str(e)}"

# --- INTERFACE (GRADIO) ---
with gr.Blocks(title="VIBE Local GUI") as demo:
    gr.Markdown("# 🎨 VIBE Image Editor (Local)")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image", type="numpy")
            prompt = gr.Textbox(label="Instruction", placeholder="e.g. Make it look like a pencil sketch", value="make it night time")
            
            with gr.Group():
                gr.Markdown("### Settings")
                steps = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Sample Steps")
                g_scale = gr.Slider(minimum=0.1, maximum=30, value=4.5, step=0.1, label="Guidance Scale (Text Strength)")
                i_scale = gr.Slider(minimum=0.1, maximum=30, value=1.2, step=0.1, label="Image Guidance Scale (Original Fidelity)")
                seed = gr.Slider(minimum=-1, maximum=2147483647, value=42, step=1, label="Seed (-1 for random)")

            submit_btn = gr.Button("🚀 Generate", variant="primary")
            
        with gr.Column():
            output_img = gr.Image(label="Result")
            status_text = gr.Textbox(label="Status / Save Path", interactive=False)

    # Link button to function
    submit_btn.click(
        fn=process_image,
        inputs=[input_img, prompt, steps, g_scale, i_scale, seed],
        outputs=[output_img, status_text]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
