import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# ---------------- Path setup ----------------
try:
    # Works in normal Python scripts
    REPO_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # Fallback for Colab / Jupyter (no __file__)
    REPO_ROOT = Path.cwd()

sys.path.append(str(REPO_ROOT))

from src.utils import load_json, save_json
from src.llm_utils import call_llm

# ---------------- Files & settings ----------------
CHUNKS_FILE = REPO_ROOT / "outputs/attention_chunks.json"
OUTPUT_FILE = REPO_ROOT / "outputs/concepts_relationships.json"

# ---------------- Main logic ----------------
def extract_concepts_and_relationships(chunks):
    results = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        prompt = f"""
        Extract key **concepts** and **relationships** between them from the following text:

        {chunk['text']}

        Respond in JSON format with:
        - "concepts": [list of concepts]
        - "relationships": [list of relationships]
        """
        response = call_llm(prompt)
        try:
            data = json.loads(response)
        except Exception:
            data = {"concepts": [], "relationships": []}
        results.append({
            "chunk_id": chunk["id"],
            "concepts": data.get("concepts", []),
            "relationships": data.get("relationships", [])
        })
    return results


if __name__ == "__main__":
    print(f"📂 Loading chunks from: {CHUNKS_FILE}")
    chunks = load_json(CHUNKS_FILE)

    print("🔍 Extracting concepts and relationships...")
    results = extract_concepts_and_relationships(chunks)

    print(f"💾 Saving output to: {OUTPUT_FILE}")
    save_json(results, OUTPUT_FILE)
    print("✅ Done")
