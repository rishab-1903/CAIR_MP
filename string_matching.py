import gradio as gr
import json
import store_caption
from sentence_transformers import SentenceTransformer, util
import speech_recognition as sr
from deep_translator import GoogleTranslator
import nltk
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# ✅ Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Load BERT model for semantic similarity
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Global storage for captions and folder ID
captions = {}
image_links = {}
folder_id = ""

def preprocess_text(text):
    """Lowercase, remove punctuation, and stopwords for better matching."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

def initialize_captions(input_folder_id):
    """Fetch captions and store in JSON file."""
    global captions, image_links, folder_id
    try:
        num_captions = store_caption.fetch_and_store_captions(input_folder_id)
        with open("captions.json", "r") as f:
            data = json.load(f)
        folder_id = data.get("folder_id", "")
        captions = data.get("images", {})
        image_links = data.get("image_links", {})
        return f"✅ {num_captions} captions stored! Ready for searching."
    except Exception as e:
        return f"❌ Error initializing captions: {e}"

def compute_tfidf_similarity(query, captions_list):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query] + captions_list)
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()

def compute_fuzzy_matching(query, captions_list):
    return np.array([fuzz.partial_ratio(query, caption) for caption in captions_list]) / 100.0

def search_captions(query):
    """Find top matching captions using the Multilevel Approach."""
    if not captions:
        return "❌ No captions available. Fetch them first!"
    query = preprocess_text(query)
    query_embedding = bert_model.encode(query, convert_to_tensor=True)
    captions_list = list(captions.values())
    image_names = list(captions.keys())

    bert_scores = [util.pytorch_cos_sim(query_embedding, bert_model.encode(caption, convert_to_tensor=True)).item() for caption in captions_list]
    tfidf_scores = compute_tfidf_similarity(query, captions_list)
    fuzzy_scores = compute_fuzzy_matching(query, captions_list)
    final_scores = 0.5 * np.array(bert_scores) + 0.3 * np.array(tfidf_scores) + 0.2 * np.array(fuzzy_scores)
    
    top_indices = np.argsort(final_scores)[::-1][:3]
    results = []
    for idx in top_indices:
        img_name = image_names[idx]
        caption = captions_list[idx]
        image_url = image_links.get(img_name, "#")
        results.append(f"🔹 *{caption}* | [View Image]({image_url})")
    
    # ✅ Plotting Graph
    plt.figure(figsize=(8, 5))
    plt.bar([captions_list[i] for i in top_indices], [final_scores[i] for i in top_indices], color=['blue', 'green', 'red'])
    plt.xlabel("Captions")
    plt.ylabel("Relevance Score")
    plt.title("Top 3 Search Results")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return "\n\n".join(results) if results else "❌ No relevant images found."

def speech_to_text(selected_language):
    """Convert speech to text and translate to English."""
    recognizer = sr.Recognizer()
    lang_code = "hi-IN" if selected_language == "Hindi" else "te-IN"
    with sr.Microphone() as source:
        print(f"🎙 Speak in {selected_language} ({lang_code})... (Auto-stop enabled)")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio, language=lang_code)
            translated_text = GoogleTranslator(source="auto", target="en").translate(text)
            return translated_text
        except sr.WaitTimeoutError:
            return "⏳ No speech detected. Try again."
        except Exception as e:
            return f"❌ Error: {e}"

# ✅ Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 📷 Context Aware Image Retrieval System - Voice & Text Search")
    with gr.Row():
        folder_id_input = gr.Textbox(label="Google Drive Folder ID", placeholder="Enter folder ID")
        fetch_button = gr.Button("📥 Fetch Captions")
    status_output = gr.Textbox(label="Status", interactive=False)
    fetch_button.click(initialize_captions, inputs=[folder_id_input], outputs=[status_output])
    with gr.Row():
        query_input = gr.Textbox(label="🔍 Enter search query")
        language_selector = gr.Dropdown(["Hindi", "Telugu"], label="🎙 Select Speech Language")
        voice_button = gr.Button("🎙 Speak")
    search_button = gr.Button("Search")
    search_output = gr.Markdown()
    voice_button.click(speech_to_text, inputs=[language_selector], outputs=[query_input])
    search_button.click(search_captions, inputs=[query_input], outputs=[search_output])
app.launch()