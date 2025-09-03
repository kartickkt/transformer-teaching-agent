import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------- Settings ----------------
INPUT_FILE = "outputs/phase2_concepts.json"
OUTPUT_FILE = "outputs/phase2_relationships.json"

# ---------------- Load Model ----------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ---------------- Prompt Template ----------------
PROMPT_TEMPLATE = """
You are an information extraction system.
From the following list of concepts, extract semantic relationships between them.

Only output valid JSON in this format:
[
  {{"subject": "concept1", "relation": "relation_type", "object": "concept2"}},
  ...
]

Concepts:
{concepts}

JSON:
"""

# ---------------- Load Concepts ----------------
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

results = []

# ---------------- Extract Relationships ----------------
for i, ex in enumerate(tqdm(data, desc="Extracting relations")):
    concepts = ", ".join(ex["concepts"])
    prompt = PROMPT_TEMPLATE.format(concepts=concepts)

    response = generator(
        prompt,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=False
    )[0]["generated_text"]

    # Debug: print first 3 raw responses
    if i < 3:
        print("\n========================")
        print(f"Chunk {i} Concepts: {concepts}")
        print("\n--- Raw Response ---\n", response)
        print("========================\n")

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

# ---------------- Save ----------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Relationships extracted and saved to {OUTPUT_FILE}")
