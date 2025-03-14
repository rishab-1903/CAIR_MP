import gradio as gr
import json
import store_caption
from sentence_transformers import SentenceTransformer, util
import speech_recognition as sr
from deep_translator import GoogleTranslator

# ✅ Load BERT model for similarity search
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Global storage for captions and folder ID
captions = {}
image_links = {}
folder_id = ""

def initialize_captions(input_folder_id):
    """Fetch captions and store in JSON file."""
    global captions, image_links, folder_id

    try:
        # ✅ Fetch and store captions from Google Drive
        num_captions = store_caption.fetch_and_store_captions(input_folder_id)

        # ✅ Load stored captions & image links from JSON
        with open("captions.json", "r") as f:
            data = json.load(f)  

        # ✅ Extract folder_id, captions, and image links from JSON
        folder_id = data.get("folder_id", "")
        captions = data.get("images", {})  
        image_links = data.get("image_links", {})  

        return f"✅ {num_captions} captions stored! Ready for searching."

    except Exception as e:
        return f"❌ Error initializing captions: {e}"

def search_captions(query):
    """Find top matching captions using BERT similarity search."""
    if not captions:
        return "❌ No captions available. Fetch them first!"

    # ✅ Compute query embedding
    query_embedding = bert_model.encode(query, convert_to_tensor=True)

    # ✅ Compute similarity scores
    similarities = []
    for img_name, caption in captions.items():
        caption_embedding = bert_model.encode(caption, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, caption_embedding).item()
        similarities.append((img_name, caption, score))

    # ✅ Get top match
    top_matches = sorted(similarities, key=lambda x: x[2], reverse=True)[:1]

    # ✅ Prepare results
    results = []
    for img_name, caption, score in top_matches:
        image_url = image_links.get(img_name, "#")
        results.append(f"🔹 **{caption}** | [View Image]({image_url})")

    return "\n\n".join(results)  

def speech_to_text(selected_language):
    """Convert speech to text and translate to English based on selected language."""
    recognizer = sr.Recognizer()
    lang_code = "hi-IN" if selected_language == "Hindi" else "te-IN"

    with sr.Microphone() as source:
        print(f"🎙 Speak in {selected_language} ({lang_code})... (Auto-stop enabled)")
        recognizer.adjust_for_ambient_noise(source)

        try:
            # Stop listening after 5 seconds of silence or continuous speech
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

            # Convert speech to text using the selected language
            text = recognizer.recognize_google(audio, language=lang_code)
            print(f"Recognized Text: {text}")

            # Translate to English
            translated_text = GoogleTranslator(source="auto", target="en").translate(text)
            print(f"Translated Text: {translated_text}")
            return translated_text

        except sr.WaitTimeoutError:
            return "⏳ No speech detected. Try again."

        except Exception as e:
            return f"❌ Error: {e}"

# ✅ Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 📷 Context Aware Image Retrieval System - Voice & Text Search")

    # ✅ Section 1: Fetch Captions
    with gr.Row():
        folder_id_input = gr.Textbox(label="Google Drive Folder ID", placeholder="Enter folder ID")
        fetch_button = gr.Button("📥 Fetch Captions")

    status_output = gr.Textbox(label="Status", interactive=False)
    fetch_button.click(initialize_captions, inputs=[folder_id_input], outputs=[status_output])

    # ✅ Section 2: Search Captions
    with gr.Row():
        query_input = gr.Textbox(label="🔍 Enter search query")
        language_selector = gr.Dropdown(["Hindi", "Telugu"], label="🎙 Select Speech Language")
        voice_button = gr.Button("🎙 Speak")

    search_button = gr.Button("Search")
    search_output = gr.Markdown()

    # ✅ Voice Search - Convert Speech to Text based on Selected Language
    voice_button.click(speech_to_text, inputs=[language_selector], outputs=[query_input])

    # ✅ Perform Search
    search_button.click(search_captions, inputs=[query_input], outputs=[search_output])

app.launch()
