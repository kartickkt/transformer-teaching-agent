# src/02_initial_concept_extract.py

import json
from tqdm import tqdm
import langextract as lx
from dotenv import load_dotenv

# Load API key
load_dotenv()

# ---------------- Files ----------------
CHUNKS_FILE = "outputs/attention_chunks.json"
OUTPUT_FILE = "outputs/02_initial_concept_extract_output.json"

# ---------------- Load chunks ----------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ---------------- Prompt ----------------
prompt = """
Extract concepts from the text.
- A concept is a key technical term or entity.
- Use exact wording from the text.
- Output only the concepts, no relationships.
"""

# ---------------- Few-shot example ----------------
examples = [
    lx.data.ExampleData(
        text="The Transformer uses multi-head attention.",
        extractions=[
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="Transformer",
            ),
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="multi-head attention",
            ),
        ]
    )
]

# ---------------- Extract concepts ----------------
all_concepts = []

for chunk in tqdm(chunks, desc="Extracting concepts"):
    extraction = lx.extract(
        text_or_documents=chunk["text"],
        prompt_description=prompt,
        examples=examples,
        model_id="gpt-4o-mini",  # OpenAI model
    )

    # Simplify to just the text values
    concepts = [e.extraction_text for e in extraction.extractions if e.extraction_class == "concept"]
    all_concepts.extend(concepts)

# ---------------- Save results ----------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_concepts, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(all_concepts)} concepts. Saved to {OUTPUT_FILE}")
