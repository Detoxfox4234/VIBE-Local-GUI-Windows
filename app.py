import gradio as gr
import torch
from PIL import Image
from vibe.editor import ImageEditor
import os
import time
import random
import sys

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = "VIBE_Model" 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
print("Starting GUI...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device}")

checkpoint = MODEL_PATH if os.path.exists(MODEL_PATH) else "iitolstykh/VIBE-Image-Edit"

print(f"Loading model from: {checkpoint}")
try:
    editor = ImageEditor(checkpoint_path=checkpoint)
    _ = editor.pipe.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have installed: pip install git+https://github.com/ai-forever/VIBE")
    exit()

print("Ready! Launching interface...")

# --- FUNCTIONS ---

def shutdown_server():
    """Stops the process and releases VRAM immediately."""
    print("Shutting down... Releasing VRAM.")
    os._exit(0)

def process_image(input_image, instruction, steps, guidance_scale, image_guidance_scale, seed):
    if input_image is None:
        return None, "Error: No image provided."
    
    image = Image.fromarray(input_image).convert("RGB")
    
    if seed == -1:
        seed = random.randint(0, 2147483647)
    
    print(f"Processing: '{instruction}' | Seed: {seed} | Steps: {steps}")

    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    try:
        edited_image = editor.pipe(
            prompt=instruction,
            conditioning_image=image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            image_guidance_scale=float(image_guidance_scale),
            generator=generator
        ).images[0]
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"VIBE_{timestamp}_seed{seed}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        edited_image.save(save_path, quality=100)
        return edited_image, f"Success! Image saved to: {save_path}"
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return None, f"Error: {str(e)}"

# --- GUI LAYOUT ---
with gr.Blocks(title="VIBE Local GUI") as demo:
    gr.Markdown("# 🎨 VIBE Image Editor (Local)")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image", type="numpy")
            # --- IMPROVED TEXTBOX ---
            prompt = gr.Textbox(
                label="Instruction", 
                placeholder="e.g. Change the background to a snowy mountain or make it look like a pencil sketch",
                lines=2
            )
            
            with gr.Group():
                gr.Markdown("### Advanced Settings")
                steps = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Sample Steps")
                g_scale = gr.Slider(minimum=0.1, maximum=30, value=4.5, step=0.1, label="Guidance Scale (Text Influence)")
                i_scale = gr.Slider(minimum=0.1, maximum=30, value=1.2, step=0.1, label="Image Guidance Scale (Original Preservation)")
                seed = gr.Slider(minimum=-1, maximum=2147483647, value=42, step=1, label="Seed (-1 for random)")

            with gr.Row():
                submit_btn = gr.Button("🚀 Generate", variant="primary")
                exit_btn = gr.Button("🛑 Shutdown Server", variant="stop")
            
        with gr.Column():
            output_img = gr.Image(label="Result")
            status_text = gr.Textbox(label="Status & File Path", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### 🎧 Support the Creator")
            gr.Markdown("If you find this tool helpful, feel free to support me by following my [Spotify](https://open.spotify.com/artist/7EdK2cuIo7xTAacutHs9gv?si=BgHnU-sxRmOxfHHsqMnlqg) profile. Every listen counts!")

    # Event Handlers
    submit_btn.click(
        fn=process_image,
        inputs=[input_img, prompt, steps, g_scale, i_scale, seed],
        outputs=[output_img, status_text]
    )
    
    exit_btn.click(fn=shutdown_server)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
