import time
import torch
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import concurrent.futures
import multiprocessing

# Detect GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BLIP Processor and Model on GPU
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Determine the optimal number of workers
MAX_WORKERS = min(8, multiprocessing.cpu_count())

def generate_caption(img):
    """Generate a caption for a single image using GPU."""
    if isinstance(img, np.ndarray):  
        img = Image.fromarray(img)  

    with torch.no_grad(): 
        inputs = processor(img, return_tensors="pt").to(device)  
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

def generate_captions_multiprocessing(images, max_workers=MAX_WORKERS, batch_size=8):
    """Generate captions using optimized multiprocessing with GPU support."""
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(generate_caption, images, chunksize=batch_size))
    
    return results