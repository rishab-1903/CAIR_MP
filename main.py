from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import concurrent.futures 

# Load BLIP Processor and Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(img):
    """Generate a caption for a single image."""
    if isinstance(img, np.ndarray):  
        img = Image.fromarray(img)  # Convert NumPy array to PIL Image

    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_captions_parallel(images, max_workers=4):
    """Generate captions for multiple images in parallel."""
    captions = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(generate_caption, images)  # Parallel execution

    captions = list(results)  # Convert generator to list
    return captions