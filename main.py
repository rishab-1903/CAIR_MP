from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP Processor and Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(img):
    if isinstance(img, np.ndarray):  
        img = Image.fromarray(img)  # Convert NumPy array to PIL Image

    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption