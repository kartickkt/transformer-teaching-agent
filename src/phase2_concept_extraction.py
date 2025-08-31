import json
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ---------------- Paths ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # assuming src/ is current folder
CHUNKS_FILE = REPO_ROOT / "outputs/attention_chunks.json"
OUTPUT_FILE = REPO_ROOT / "outputs/concepts.json"

# ---------------- Load chunks ----------------
with open(CHUNKS_FILE, "r") as f:
    chunks = json.load(f)

# ---------------- Load Mistral model ----------------
model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # replace with your local path if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True  # you can change to 8bit if needed
)
hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.0,
)

# ---------------- Helper ----------------
def extract_concepts(text):
    prompt = (
        f"Extract the main concepts from the following text as a JSON list of strings:\n\n{text}\n\nConcepts:"
    )
    output = hf_pipe(prompt)[0]["generated_text"]
    try:
        # Attempt to parse JSON directly from model output
        start = output.find("[")
        end = output.rfind("]") + 1
        concepts = json.loads(output[start:end])
    except Exception:
        concepts = []
    return concepts

# ---------------- Run extraction ----------------
results = []
for chunk in tqdm(chunks, desc="Extracting concepts"):
    text = chunk if isinstance(chunk, str) else chunk.get("text", "")
    concepts = extract_concepts(text)
    results.append({"text": text, "concepts": concepts})

# ---------------- Save results ----------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved extracted concepts to {OUTPUT_FILE}")
