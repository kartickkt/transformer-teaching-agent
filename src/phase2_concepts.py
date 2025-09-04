import json
from pathlib import Path
from dotenv import load_dotenv
import os
from tqdm import tqdm

import langextract as lx

# ---------------- Setup ----------------
load_dotenv()  # load OPENAI_API_KEY from .env
repo_root = Path(__file__).resolve().parents[1]

INPUT_FILE = repo_root / "outputs" / "attention_chunks.json"
OUTPUT_FILE = repo_root / "outputs" / "phase2_concepts.json"

# ---------------- Prompt & Examples ----------------
prompt = """Extract key concepts and relationships from the text.
- List all important technical terms as 'concepts'.
- Identify relationships between concepts as subject-predicate-object triples.
- Use only terms from the text."""

# Optional: You can add a few-shot example to guide extraction
examples = [
    lx.data.ExampleData(
        text="The Transformer uses self-attention and multi-head attention.",
        extractions=[
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="Transformer"
            ),
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="self-attention"
            ),
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="multi-head attention"
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Transformer uses self-attention",
                attributes={"type": "uses"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Transformer uses multi-head attention",
                attributes={"type": "uses"}
            ),
        ]
    )
]

# ---------------- Load Chunks ----------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

results = []

# ---------------- Run Extraction ----------------
for i, chunk in enumerate(tqdm(chunks, desc="Extracting concepts/relationships")):
    text = chunk["text"]

    extraction = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    results.append({
        "chunk_id": i,
        "text": text,
        "entities": [e.to_dict() for e in extraction.entities],
        "relationships": [r.to_dict() for r in extraction.relationships],
    })

# ---------------- Save Results ----------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved {len(results)} chunks with concepts/relationships → {OUTPUT_FILE}")
