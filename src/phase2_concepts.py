import json
from pathlib import Path
from dotenv import load_dotenv
import os
from tqdm import tqdm

from langextract import Extractor
from langextract.schema import Extraction, Entity, Relationship

# ---------------- Setup ----------------
load_dotenv()  # load OPENAI_API_KEY from .env
repo_root = Path(__file__).resolve().parents[1]

INPUT_FILE = repo_root / "outputs" / "attention_chunks.json"
OUTPUT_FILE = repo_root / "outputs" / "phase2_concepts.json"

# ---------------- LangExtract Config ----------------
extractor = Extractor(
    provider="openai",
    model="gpt-4o-mini",  # cheapest reliable option
)

# Define schema (concepts = entities, links = relationships)
schema = Extraction(
    entities=[
        Entity(name="Concept", description="Any important technical term or concept"),
    ],
    relationships=[
        Relationship(
            name="related_to",
            description="Relationship between two concepts (e.g., 'uses', 'depends on', 'enables')",
            source="Concept",
            target="Concept",
        ),
    ],
)

# ---------------- Load Chunks ----------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

results = []

# ---------------- Run Extraction ----------------
for i, chunk in enumerate(tqdm(chunks, desc="Extracting concepts/relationships")):
    text = chunk["text"]

    # Run LangExtract
    extraction = extractor.extract(
        schema=schema,
        text=text,
        instructions="Extract key concepts and relationships as subject-predicate-object triples. "
                     "Be concise, use only terms from the text.",
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
