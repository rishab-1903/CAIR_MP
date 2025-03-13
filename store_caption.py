from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
import numpy as np
import io
import main as b  # Import BLIP-based captioning function
import json
import logging

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ Statically configure the API key (replace with your local file path)
API_KEY_PATH = "concise-reserve-449207-p2-3a505c75d108.json"

# ✅ Load Google Drive API credentials (no UI upload needed)
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
credentials = service_account.Credentials.from_service_account_file(API_KEY_PATH, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

def list_images_in_folder(folder_id):
    """List all image files in a Google Drive folder."""
    query = f"'{folder_id}' in parents and mimeType contains 'image/'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    return results.get('files', [])

def fetch_image_from_drive(file_id):
    """Fetch image content from Google Drive as a NumPy array."""
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    logging.info(f"Image downloaded in memory from Drive: {file_id}")
    fh.seek(0)
    
    pil_image = Image.open(fh)  # Convert to PIL Image
    return np.array(pil_image)  # Convert to NumPy array before returning

def fetch_and_store_captions(folder_id):
    """Fetch captions for images in a folder and save them to captions.json."""
    logging.info("Fetching image list from Drive...")
    images = list_images_in_folder(folder_id)

    captions = {}

    for image in images:
        try:
            file_id = image['id']
            file_name = image['name']

            # ✅ Fetch the image as a NumPy array
            logging.info(f"Processing image: {file_name}")
            np_image = fetch_image_from_drive(file_id)

            # ✅ Generate the caption using BLIP model
            caption = b.generate_caption(np_image)  # Pass NumPy array
            captions[file_name] = caption

            logging.info(f"Caption generated for {file_name}: {caption}")

        except Exception as e:
            logging.error(f"Error processing image {file_name}: {e}")

    # ✅ Save captions to captions.json
    with open("captions.json", "w") as f:
        json.dump(captions, f, indent=4)

    logging.info("Captions saved to captions.json.")
    return len(captions), {img["name"]: f"https://drive.google.com/file/d/{img['id']}/view" for img in images}