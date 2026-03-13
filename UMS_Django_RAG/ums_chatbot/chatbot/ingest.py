import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Persistent Chroma storage
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "vector_db", "chroma")

client = chromadb.PersistentClient(path=CHROMA_PATH)

model = SentenceTransformer("all-MiniLM-L6-v2")


# ─── Text Extraction ────────────────────────────────────────

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file, page by page."""
    reader = PdfReader(file_path)
    pages = []
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages)


def extract_text_from_txt(file_path):
    """Extract text from a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(file_path):
    """Extract text from supported file types."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext in [".doc", ".docx"]:
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            print(f"  ⚠ Skipping {file_path} — install python-docx for .docx support")
            return ""
        except Exception as e:
            print(f"  ⚠ Could not read {file_path}: {e}")
            return ""
    elif ext == ".json":
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    else:
        print(f"  ⚠ Unsupported file type: {ext} ({file_path})")
        return ""


# ─── Text Cleaning ───────────────────────────────────────────

def clean_text(text):
    """Clean extracted text by removing noise and normalizing whitespace."""
    # Remove excessive whitespace/newlines but preserve paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page numbers like "1 | Page", "Page 1 of 10"
    text = re.sub(r'\d+\s*\|\s*Page', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    # Remove excessive spaces
    text = re.sub(r'[ \t]{3,}', ' ', text)
    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # Remove empty lines in sequence (keep max 1 blank line)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─── Sentence-Aware Chunking ─────────────────────────────────

def smart_chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into chunks that respect sentence boundaries.
    
    - Larger chunk_size (1000 chars) gives more context per chunk
    - Overlap (200 chars) ensures no info lost at boundaries
    - Splits at sentence boundaries (., !, ?) to avoid mid-sentence cuts
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence exceeds chunk_size, save current chunk and start new one
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Create overlap: take the last `overlap` characters of the current chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            # Find the start of the first complete sentence in overlap
            overlap_sentence_start = overlap_text.find('. ')
            if overlap_sentence_start != -1:
                current_chunk = overlap_text[overlap_sentence_start + 2:]
            else:
                current_chunk = ""
        
        current_chunk += " " + sentence if current_chunk else sentence
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks (likely noise)
    chunks = [c for c in chunks if len(c) > 50]
    
    return chunks


# ─── Ingestion ────────────────────────────────────────────────

def ingest_file(file_path, collection_name):
    """Ingest a single file into the vector database with smart chunking."""
    print(f"  📄 Processing: {os.path.basename(file_path)}")

    text = extract_text(file_path)

    if not text.strip():
        print(f"  ⚠ No text extracted from {file_path}, skipping.")
        return 0

    # Clean the text first
    text = clean_text(text)

    # Use smart sentence-aware chunking
    chunks = smart_chunk_text(text, chunk_size=400, overlap=50)

    if not chunks:
        print(f"  ⚠ No valid chunks from {file_path}, skipping.")
        return 0

    collection = client.get_or_create_collection(name=collection_name)

    file_id = os.path.splitext(os.path.basename(file_path))[0]
    # Sanitize file_id for use as ChromaDB ID
    file_id = re.sub(r'[^a-zA-Z0-9_-]', '_', file_id)

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()

        # Add source metadata for traceability
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source_file": os.path.basename(file_path),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }],
            ids=[f"{collection_name}_{file_id}_{i}"]
        )

    print(f"  ✅ Ingested {len(chunks)} chunks into '{collection_name}' collection.")
    return len(chunks)


def ingest_folder(folder_path, collection_name):
    """Ingest all supported files in a folder into a collection."""
    print(f"\n📁 Ingesting folder: {folder_path} → collection: '{collection_name}'")

    if not os.path.exists(folder_path):
        print(f"  ❌ Folder not found: {folder_path}")
        return

    total_chunks = 0
    supported_extensions = {".pdf", ".txt", ".doc", ".docx", ".json"}

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_extensions:
            file_path = os.path.join(folder_path, filename)
            total_chunks += ingest_file(file_path, collection_name)

    print(f"📊 Total chunks in '{collection_name}': {total_chunks}")


def clear_all_collections():
    """Delete all existing collections to start fresh."""
    print("🗑️  Clearing all existing collections...")
    for col in client.list_collections():
        print(f"  Deleting: {col.name}")
        client.delete_collection(col.name)
    print("  ✅ All collections cleared.")


def ingest_all():
    """Ingest all data folders into their respective collections."""
    data_dir = os.path.join(BASE_DIR, "data")

    # Clear old data first for a clean re-ingest
    clear_all_collections()

    # Map folder names to collection names (matching intents in intent.py)
    # A single folder can map to MULTIPLE collections for better coverage
    folder_to_collections = {
        "academics": ["courses", "placement"],  # has placement & course docs
        "admissions": ["admission", "fees"],     # has admission & fee docs
        "exams": ["exam"],
        "hostel": ["hostel", "emergency"],       # has emergency numbers too
        "users": ["general"],
    }

    print("=" * 60)
    print("  UMS RAG CHATBOT — DATA INGESTION (Improved)")
    print("=" * 60)

    for folder_name, collection_names in folder_to_collections.items():
        folder_path = os.path.join(data_dir, folder_name)
        for collection_name in collection_names:
            ingest_folder(folder_path, collection_name)

    # Print summary
    print("\n" + "=" * 60)
    print("  📊 FINAL SUMMARY")
    print("=" * 60)
    for col in client.list_collections():
        print(f"  {col.name}: {col.count()} chunks")

    print("\n" + "=" * 60)
    print("  ✅ ALL INGESTION COMPLETE!")
    print("=" * 60)


# Legacy function for backward compatibility
def ingest_pdf(file_path, collection_name):
    """Ingest a single PDF (legacy function)."""
    ingest_file(file_path, collection_name)
    print("Ingestion complete!")
