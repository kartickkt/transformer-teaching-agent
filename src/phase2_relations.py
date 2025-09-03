# src/phase2_relations.py

import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ============== Settings ==============
INPUT_FILE = Path("outputs/phase2_concept.json")
OUTPUT_FILE = Path("outputs/phase2_relations.json")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# ============== Load Model ==============
print("Loading Mistral model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ============== Prompt Template ==============
PROMPT_TEMPLATE = """
You are an information extraction system.
Given a list of concepts, extract relationships as JSON triplets.

Concepts:
{concepts}

Output format (JSON list):
[
  {{"subject": "...", "relation": "...", "object": "..."}},
  ...
]
"""

# ============== Load Phase 1 Concepts ==============
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

results = []

# ============== Process Chunks ==============
for ex in tqdm(data, desc="Extracting relations"):
    concepts = ", ".join(ex["concepts"])
    prompt = PROMPT_TEMPLATE.format(concepts=concepts)

    response = generator(
        prompt,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=False
    )[0]["generated_text"]

    # Try to parse JSON
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        triplets = json.loads(response[start:end])
    except Exception:
        triplets = []

    results.append({
        "chunk_id": ex["chunk_id"],
        "relationships": triplets
    })

# ============== Save Output ==============
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved relations to {OUTPUT_FILE}")
