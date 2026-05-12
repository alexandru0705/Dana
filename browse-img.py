import os
import json
from mlx_vlm import load, generate
from PIL import Image

MODEL_PATH = "/Users/alexandrudumitru/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality"
IMAGE_DIR = "images"
OUTPUT_DATA = "products.json"

print("Initializing Qwen 3.6...")
model, processor = load(MODEL_PATH)
catalog = []

for i, filename in os.listdir(IMAGE_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGE_DIR, filename)
        category = filename.split('-')[0]
        
        try:
            # 1. Strip metadata to fix the 0xb9 error
            raw_image = Image.open(path).convert("RGB")
            prompt = f"USER: <|vision_start|><|image_pad|><|vision_end|>Describe this {category} for Dana's shop.\nASSISTANT:"

            # 2. Disable streaming (verbose=False) and set temp=0.0 to stop crashes
            result = generate(model, processor, prompt, [raw_image], temp=0.0, max_tokens=300, verbose=False)
            
            catalog.append({
                "id": i + 1,
                "filename": filename, 
                "category": category, 
                "description": result.text.strip()
            })
            print(f"Done: {filename}")
        except Exception as e:
            print(f"Skipping {filename} due to model error: {e}")

with open(OUTPUT_DATA, "w", encoding='utf-8') as f:
    json.dump(catalog, f, indent=4)
print(f"Success! Data saved in {OUTPUT_DATA}")