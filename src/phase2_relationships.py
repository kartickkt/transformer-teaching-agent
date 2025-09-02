# scripts/phase2_relationships.py

import json
import os
from pathlib import Path
import sys
from tqdm import tqdm

# ---------------- Path setup for Colab / local ----------------
repo_root = Path(__file__).resolve().parents[2]  # go up two levels from scripts/
sys.path.append(str(repo_root))

from src.utils import load_json, save_json
from src.llm_utils import call_llm

# ---------------- Files & settings ----------------
INPUT_FILE = "outputs/phase2_concepts.json"
OUTPUT_FILE = "outputs/phase2_relationships.json"
BATCH_SIZE = 30  # number of concepts per LLM call


# ---------------- Prompt ----------------
def make_prompt(concepts):
    return f"""
You are a scientific assistant. Given the following list of concepts, extract meaningful relationships
between them and return only a JSON list in the format:

[
  {{
    "subject": "ConceptA",
    "relation": "is related to",
    "object": "ConceptB"
  }}
]

Guidelines:
- Use ONLY the provided concepts (no hallucinations).
- Relations must be concise but meaningful (e.g., "is part of", "depends on", "interacts with").
- Avoid duplicates or reversed duplicates (A→B, B→A unless both directions have distinct meanings).
- Ensure every relation links two distinct concepts.

Concept list:
{json.dumps(concepts, indent=2)}
"""


# ---------------- Main logic ----------------
def extract_relationships(concepts):
    relationships = []

    for i in tqdm(range(0, len(concepts), BATCH_SIZE)):
        batch = concepts[i : i + BATCH_SIZE]
        prompt = make_prompt(batch)
        response = call_llm(prompt)

        try:
            rels = json.loads(response)
            if isinstance(rels, list):
                relationships.extend(rels)
        except json.JSONDecodeError:
            print("⚠️ Failed to parse JSON for batch, skipping...")

    return relationships


def main():
    print("📂 Loading concepts...")
    concepts = load_json(INPUT_FILE)

    # Handle dict (if saved with keys) vs list
    if isinstance(concepts, dict) and "concepts" in concepts:
        concepts = concepts["concepts"]

    print(f"✅ Loaded {len(concepts)} concepts")

    print("🔗 Extracting relationships...")
    relationships = extract_relationships(concepts)

    print(f"✅ Extracted {len(relationships)} relationships")

    save_json(relationships, OUTPUT_FILE)
    print(f"💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
