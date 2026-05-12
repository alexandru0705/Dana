import os
import json
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config

# 1. Setup paths
MODEL_PATH = "/Users/alexandrudumitru/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality"
IMAGE_DIR = "images"
OUTPUT_DATA = "products.json"

# 2. Load the model onto your M5 Pro
print("Initializing Qwen 3.6...")
model, processor = load(MODEL_PATH)
config = load_config(MODEL_PATH)

catalog = []

# 3. Dynamic Scanning
print(f"Scanning folder: {IMAGE_DIR}")
for filename in os.listdir(IMAGE_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGE_DIR, filename)
        
        # We use your naming convention to help the model
        category = filename.split('-')[0] 
        
        prompt = f"This is an artisan piece by Dana in Paris. It's in the category '{category}'. Describe its style and material for a shop listing."
        
        description = generate(model, processor, prompt, [path], verbose=False)
        
        catalog.append({
            "filename": filename,
            "category": category,
            "description": description.strip()
        })
        print(f"Done: {filename}")

# 4. Save the result
with open(OUTPUT_DATA, "w") as f:
    json.dump(catalog, f, indent=4)

print(f"\nSuccess! Your dynamic data is saved in {OUTPUT_DATA}")