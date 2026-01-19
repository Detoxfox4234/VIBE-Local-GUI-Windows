import gradio as gr
import torch
from PIL import Image
from vibe.editor import ImageEditor
import os
import time
import random

# --- KONFIGURATION ---
# Wir nutzen relative Pfade, damit es auf jedem PC funktioniert
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# Das Modell wird im Cache oder lokal gesucht
MODEL_PATH = "VIBE_Model" 

# Ordner erstellen
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODELL LADEN ---
print("Starte GUI...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Gerät erkannt: {device}")

# Checkpoint Pfad Logik:
# Wenn der Ordner "VIBE_Model" existiert, nimm ihn (Offline Modus).
# Wenn nicht, lädt er es neu von HuggingFace (Online Modus).
checkpoint = MODEL_PATH if os.path.exists(MODEL_PATH) else "iitolstykh/VIBE-Image-Edit"

print(f"Lade Modell von: {checkpoint}")
try:
    editor = ImageEditor(checkpoint_path=checkpoint)
    # Kleiner Hack: Wir greifen direkt auf die Pipe zu, um Zugriff auf Generator/Steps zu haben
    # Das initialisiert das Modell auf der GPU
    _ = editor.pipe.to(device)
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    print("Stelle sicher, dass du 'pip install git+https://github.com/ai-forever/VIBE' ausgeführt hast.")
    exit()

print("Bereit! Starte Oberfläche...")

# --- DIE FUNKTION ---
def verarbeite_bild(input_image, instruction, steps, guidance_scale, image_guidance_scale, seed):
    if input_image is None:
        return None
    
    image = Image.fromarray(input_image).convert("RGB")
    
    if seed == -1:
        seed = random.randint(0, 2147483647)
    
    print(f"Generiere: '{instruction}' | Seed: {seed} | Steps: {steps}")

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
        return edited_image, save_path
        
    except Exception as e:
        return None, f"Fehler: {str(e)}"

# --- GUI ---
with gr.Blocks(title="VIBE Local GUI") as demo:
    gr.Markdown("# 🎨 VIBE Image Editor (Local)")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image", type="numpy")
            prompt = gr.Textbox(label="Instruction", placeholder="e.g. Make it look like a pencil sketch", value="make it night time")
            
            with gr.Group():
                steps = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Sample Steps")
                g_scale = gr.Slider(minimum=0.1, maximum=30, value=4.5, step=0.1, label="Guidance Scale (Text Strength)")
                i_scale = gr.Slider(minimum=0.1, maximum=30, value=1.2, step=0.1, label="Image Guidance Scale (Original Fidelity)")
                seed = gr.Slider(minimum=-1, maximum=2147483647, value=42, step=1, label="Seed (-1 for random)")

            submit_btn = gr.Button("🚀 Generate", variant="primary")
            
        with gr.Column():
            output_img = gr.Image(label="Result")
            status_text = gr.Textbox(label="Status / Save Path", interactive=False)

    submit_btn.click(
        fn=verarbeite_bild,
        inputs=[input_img, prompt, steps, g_scale, i_scale, seed],
        outputs=[output_img, status_text]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
