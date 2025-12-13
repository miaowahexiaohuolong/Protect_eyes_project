import json
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= Config =================
INPUT_FILE = "dataset.json"
KB_DIR = "my_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# =======================================

def build_knowledge_base():
    print(f"🏗️ Preparing to build knowledge base...")
    
    # 1. Check data file
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File {INPUT_FILE} not found. Please run the data collection step first.")
        return

    # 2. Read JSON data (robust format handling)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    articles = []
    
    # Auto-detect common JSON structures
    # A: dict with evidence_list (current format)
    if isinstance(raw_data, dict) and "evidence_list" in raw_data:
        print("   -> Detected 'evidence_list' array...")
        articles = raw_data["evidence_list"]
        
    # B: direct list
    elif isinstance(raw_data, list):
        print("   -> Detected article list...")
        articles = raw_data
        
    # C: single article dict
    elif isinstance(raw_data, dict):
        print("   -> Detected single article...")
        articles = [raw_data]
    
    else:
        print("❌ Unrecognized data format")
        return
    
    if not articles:
        print("❌ Data is empty or extraction failed.")
        return

    print(f"📚 Loaded {len(articles)} articles, preparing embeddings...")

    # 3. Load embedding model
    print("⏳ Loading embedding model (sentence-transformers)...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    # 4. Prepare texts with tolerance for missing fields
    texts = []
    valid_articles = []
    
    for doc in articles:
        if not isinstance(doc, dict): continue
        title = doc.get('title', 'No Title')
        abstract = doc.get('abstract', '')
        if abstract and len(abstract) > 5:
            text_content = f"Title: {title}\nAbstract: {abstract}"
            texts.append(text_content)
            valid_articles.append(doc)

    if not texts:
        print("❌ No valid abstract text found.")
        return

    # 5. Compute embeddings
    print(f"⚙️ Computing embeddings for {len(texts)} items...")
    embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # 6. Normalize
    faiss.normalize_L2(embeddings)

    # 7. Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 8. Save results
    if not os.path.exists(KB_DIR):
        os.makedirs(KB_DIR)
        
    index_path = os.path.join(KB_DIR, "health.index")
    meta_path = os.path.join(KB_DIR, "health.pkl")

    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(valid_articles, f)

    print("\n" + "="*40)
    print("✅ Knowledge base built successfully!")
    print(f"📂 Output dir: {KB_DIR}/")
    print(f"   ├─ Index file: health.index")
    print(f"   └─ Data file: health.pkl")
    print("="*40)

if __name__ == "__main__":
    build_knowledge_base()