import os
import torch
import random
import time
import psutil
import gradio as gr
from PIL import Image
from huggingface_hub import snapshot_download
from vibe.editor import ImageEditor

# --- CONFIG & SETUP ---
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("⏳ Loading VIBE checkpoints (Sana1.5 + Qwen3-VL)...")
model_path = snapshot_download(repo_id="iitolstykh/VIBE-Image-Edit", repo_type="model")

# Initialize the editor
editor = ImageEditor(
    checkpoint_path=model_path,
    image_guidance_scale=1.2,
    guidance_scale=4.5,
    num_inference_steps=20,
    device=device
)

# --- FUNCTIONS ---

def shutdown_server():
    """Stops the process and releases VRAM immediately."""
    print("Shutting down... Releasing VRAM.")
    os._exit(0)

def get_system_stats():
    """Reads CPU, RAM, and VRAM usage."""
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent
    
    vram_display = "N/A"
    if torch.cuda.is_available():
        # Get free and total memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        # Calculate used memory
        used_mem = total_mem - free_mem
        
        # Convert to GB
        used_gb = used_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        percent = (used_mem / total_mem) * 100
        
        vram_display = f"{used_gb:.1f}GB / {total_gb:.1f}GB ({percent:.0f}%)"
    
    # Return HTML for nicer formatting in the UI
    return f"""
    <div style="display: flex; gap: 24px; font-family: monospace; font-size: 20px; color: #eee; font-weight: bold; align-items: center;">
        <span>🖥️ CPU: {cpu_usage}%</span>
        <span>🧠 RAM: {ram_usage}%</span>
        <span>🎮 VRAM: {vram_display}</span>
    </div>
    """

def process_image(input_image, instruction, steps, guidance_scale, image_guidance_scale, seed):
    if input_image is None:
        return None, "Error: No image provided."
    
    # Convert numpy array (Gradio) -> PIL Image
    image = Image.fromarray(input_image).convert("RGB")
    
    if seed == -1:
        seed = random.randint(0, 2147483647)
    
    print(f"Processing: '{instruction}' | Seed: {seed} | Steps: {steps}")

    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    try:
        # Use the pipeline directly from the editor object
        edited_image = editor.pipe(
            prompt=instruction,
            conditioning_image=image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            image_guidance_scale=float(image_guidance_scale),
            generator=generator
        ).images[0]
        
        # Save the resulting image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"VIBE_{timestamp}_seed{seed}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        edited_image.save(save_path, quality=100)
        return edited_image, f"Success! Image saved to: {save_path}"
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return None, f"Error: {str(e)}"

# --- GUI LAYOUT ---
with gr.Blocks(title="VIBE Local GUI", theme=gr.themes.Soft()) as demo:
    
    # HEADER ROW
    with gr.Row(variant="compact"):
        with gr.Column(scale=2):
            gr.Markdown("# 🎨 VIBE Image Editor (Local)")
        with gr.Column(scale=1):
            # This is the system monitor display (top right)
            system_stats = gr.HTML(value="Loading stats...")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image", type="numpy")
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
            
            gr.Markdown(
                """
                If you find this tool helpful, feel free to support me by following my 
                <a href="https://spotify.com" target="_blank" style="color: #1DB954; font-weight: bold; text-decoration: none;">Spotify</a> 
                profile. Every follower counts!
                """
            )

    # --- TIMER & EVENTS ---
    # Update the monitor every second (1.0s)
    timer = gr.Timer(1.0)
    timer.tick(fn=get_system_stats, outputs=system_stats)

    submit_btn.click(
        fn=process_image,
        inputs=[input_img, prompt, steps, g_scale, i_scale, seed],
        outputs=[output_img, status_text]
    )
    
    exit_btn.click(fn=shutdown_server)

if __name__ == "__main__":
    # Start the application
    demo.launch(inbrowser=True)