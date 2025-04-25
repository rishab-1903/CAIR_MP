Context-Aware Image Retrieval: Enhancing Search 
Precision in Large-Scale Image Databases Using Blip 
And Automated Captioning 

A deep learning-powered application for intelligent image search based on contextual understanding. This system combines automated image captioning with advanced semantic search techniques to deliver precise and efficient image retrieval from large-scale datasets.

---

## ✨ Features

- 📸 **Automatic Image Captioning**  
  Uses a pre-trained deep learning model to generate natural language descriptions for each image in the gallery.

- 🧠 **Hybrid Text Retrieval Engine**  
  Combines multiple retrieval techniques such as:
  - **TF-IDF** for keyword relevance
  - **Levenshtein Distance** for fuzzy string matching
  - **Sentence-BERT** for semantic similarity

- 🚀 **Batch Processing Support**  
  - Can handle image captioning in bulk using CUDA acceleration (GPU) or local CPU-based processing.
  - Efficient fetch-store mechanism allows for fast reusability without redundant processing.

- 📦 **Optimized I/O and JSON Storage**  
  - Captions and metadata are stored in well-structured JSON files.
  - Enables rapid access and minimal read/write latency even for large image collections.

- ⏱️ **Significant Time Savings**  
  The combination of optimized computation, batch pre-processing, and structured storage allows for incredibly fast query response times during image retrieval.

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/context-aware-image-retrieval.git
cd context-aware-image-retrieval
```

### 2. (Optional) Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

## 🔐 Configuration: API Key

This project requires a key JSON file (e.g., Google Cloud service credentials).

### To Set Up:

1. Visit your cloud provider's console (e.g., Firebase Console)
2. Create or access your project and generate a **Service Account Key**
3. Download the key as a `.json` file
4. Rename it to `serviceAccountKey.json` and place it in the root directory

✅ Update any hardcoded paths in the codebase if you choose a different name or location.

---

## 📂 Project Structure

```
context-aware-image-retrieval/
├── captions/                  # Generated captions stored here
├── images/                    # Your image gallery
├── model/                     # Pre-trained model files
├── serviceAccountKey.json     # API credentials
├── .gitignore                 # Git ignore file
├── .idea/                     # IDE-specific files
├── __pycache__/               # Python bytecode cache
├── captions.json              # JSON file containing generated captions
├── main.py                    # Main application script
├── requirements.txt           # Required Python packages
├── store_caption.py           # Stores generated captions
├── string_matching.py         # Hybrid string matching methods
└── README.md                  # Project documentation
```

---

## ▶️ How to Use

### 1. Add Your Images

Place your images inside the `images/` directory.

### 2. Generate Captions

```bash
python store_caption.py
```

This will:
- Extract features from each image
- Generate captions using the model
- Store results in a structured `captions/` directory in JSON format

### 3. Launch the App

#### For Gradio based apps:
```bash
python string_matching.py
```

### 4. Perform Search

Enter a text query (e.g., `"a cat sitting near a window"`)  
→ System returns the top 5 semantically relevant images.

---

## 🧠 Model & Retrieval Details

- **Encoder**: CNN (e.g., ResNet or Inception) to extract image features
- **Decoder**: RNN/Transformer model trained to produce captions  <- Combined this features using Salesforce BLIP (pretrained)
- **Retrieval Engine**:
  - Combines syntactic (TF-IDF, Levenshtein) and semantic (SBERT) methods
  - Optimized for both precision and performance
  - Flexible scoring logic configurable in `string_matching.py`

---

## ⚙️ Batch Processing & Speed Optimizations

- GPU support for large-scale caption generation using CUDA (if available)
- Pre-captioned images stored to avoid re-processing
- JSON-based structured retrieval allows for constant-time access
- Particularly suited for large datasets where repeated searches are expected

---

## 🧪 Testing

To test the system with new data:
1. Add images to `images/`
2. Re-run:
   ```bash
   python string_matching.py
   ```
3. Launch the app and query as needed

---

## 📋 Notes

- Ensure your `serviceAccountKey.json` is valid and has proper permissions
- If you deploy this app (e.g., on Heroku or Render), update storage paths accordingly
- The retrieval engine is optimized for efficient and accurate search but can be tuned by modifying the hybrid scoring logic in `string_matching.py`

---

## 📃 License

MIT License © 2025 YourName  
Free to use, modify, and distribute with attribution.
