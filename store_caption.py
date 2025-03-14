import json
import os
import logging
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
import numpy as np
import io
import main as b  # Import BLIP-based captioning function

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ Static API Key
API_KEY_PATH = "concise-reserve-449207-p2-20fabdc83541.json"

# ✅ Load Google Drive API credentials
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
credentials = service_account.Credentials.from_service_account_file(API_KEY_PATH, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

CAPTIONS_FILE = "captions.json"

def load_captions():
    """Load existing captions.json if available."""
    if os.path.exists(CAPTIONS_FILE):
        with open(CAPTIONS_FILE, "r") as f:
            return json.load(f)
    return {"folder_id": None, "images": {}}

def save_captions(data):
    """Save captions.json."""
    with open(CAPTIONS_FILE, "w") as f:
        json.dump(data, f, indent=4)

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

    logging.info(f"Image downloaded from Drive: {file_id}")
    fh.seek(0)

    pil_image = Image.open(fh)
    return np.array(pil_image)

def fetch_and_store_captions(folder_id):
    """Fetch captions for new images in a folder and update captions.json."""
    logging.info("Fetching image list from Drive...")
    images = list_images_in_folder(folder_id)

    # ✅ Load existing captions.json
    captions_data = load_captions()

    # ✅ Check if the folder has changed
    if captions_data["folder_id"] != folder_id:
        logging.info("New folder detected. Resetting captions.json.")
        captions_data = {"folder_id": folder_id, "images": {}, "image_links": {}}

    existing_images = captions_data["images"]
    existing_links = captions_data.get("image_links", {})
    new_captions = {}
    new_image_links = {}

    for image in images:
        file_id = image['id']
        file_name = image['name']
        image_link = f"https://drive.google.com/file/d/{file_id}/view"

        if file_name in existing_images:
            logging.info(f"Skipping {file_name} (already processed).")
            continue

        try:
            # ✅ Fetch the image
            logging.info(f"Processing new image: {file_name}")
            np_image = fetch_image_from_drive(file_id)

            # ✅ Generate caption using BLIP
            caption = b.generate_caption(np_image)

            # ✅ Store the caption and image link
            captions_data["images"][file_name] = caption
            captions_data["image_links"][file_name] = image_link
            new_captions[file_name] = caption
            new_image_links[file_name] = image_link

            logging.info(f"Caption generated for {file_name}: {caption}")

        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")

    # ✅ Save updated captions.json
    save_captions(captions_data)
    logging.info("Updated captions.json.")

    return len(new_captions), new_image_links  # ✅ Return updated image links