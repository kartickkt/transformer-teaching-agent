import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------- Model & tokenizer ----------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    max_new_tokens=512,
    temperature=0,   # deterministic output
    top_p=1,
    pad_token_id=tokenizer.eos_token_id
)

# ---------------- Prompt template ----------------
FEWSHOT_EXAMPLE = """
Concepts: ["Attention mechanism", "Self-attention", "Multi-head attention"]
JSON Example:
[
  {"subject": "Attention mechanism", "relation": "is_a", "object": "Self-attention"},
  {"subject": "Attention mechanism", "relation": "is_a", "object": "Multi-head attention"}
]
"""

PROMPT_TEMPLATE = """
You are an information extraction system.
From the following list of concepts, extract semantic relationships between them.

Only output valid JSON.

Concepts:
{concepts}

{fewshot_example}
"""

# ---------------- Functions ----------------
def extract_relations(chunk_concepts):
    concepts_list = json.dumps(chunk_concepts)
    prompt = PROMPT_TEMPLATE.format(concepts=concepts_list, fewshot_example=FEWSHOT_EXAMPLE)
    
    output = generator(prompt, max_new_tokens=512)[0]['generated_text']
    
    # Attempt to extract JSON from output
    try:
        json_start = output.index("[")
        json_end = output.rindex("]") + 1
        relations = json.loads(output[json_start:json_end])
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        relations = []
    
    return relations

def process_chunks(chunks, save_path="relations_output.json", batch_size=5):
    all_relations = []
    save_path = Path(save_path)
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i} to {i+len(batch)-1}...")
        
        for j, chunk in enumerate(batch):
            relations = extract_relations(chunk)
            all_relations.extend(relations)
        
        # Save intermediate results to avoid GPU waste
        save_path.write_text(json.dumps(all_relations, indent=2))
        print(f"✅ Saved intermediate results after batch {i}-{i+len(batch)-1}")
    
    return all_relations

# ---------------- Example usage ----------------
chunks = [
    ["Attention mechanism", "Self-attention", "Multi-head attention"],
    ["Transformer model", "Encoder", "Decoder", "Machine translation"],
    # ... add all your 100+ chunks here ...
]

all_relations = process_chunks(chunks, save_path="relations_output.json", batch_size=5)
print("✅ Finished processing all chunks. Relations saved to relations_output.json")
