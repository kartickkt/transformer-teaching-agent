# phase2_concept_extraction_relationships.py

import os
import json
from tqdm import tqdm
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ================== Paths ==================
REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = REPO_ROOT / "outputs/attention_chunks.json"
OUTPUT_FILE = REPO_ROOT / "outputs/concepts_relationships.json"

# ================== Load chunks ==================
with open(CHUNKS_FILE, "r") as f:
    chunks = json.load(f)  # List of strings

# ================== Load Mistral model locally ==================
model_name = "mistral-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True
)

hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.0
)

llm = HuggingFacePipeline(pipeline=hf_pipe)

# ================== Prompts ==================
concept_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Extract all the important concepts from the following text.
Provide a comma-separated list of concepts. Only list concepts, no explanations.

Text: {text}
"""
)

relationship_prompt = PromptTemplate(
    input_variables=["concepts", "text"],
    template="""
Given the following text:
{text}

And the list of concepts: {concepts}

Identify meaningful relationships between the concepts in the text.
Return output as JSON, where each concept is a key and the value is a dictionary of related concepts and their relationship.

Example:
"Self-Attention": {{
    "related_to": {{
        "RNNs": "better at handling long sequences",
        "Attention Mechanism": "is a type of"
    }}
}}

Only provide JSON output.
"""
)

concept_chain = LLMChain(llm=llm, prompt=concept_prompt)
relationship_chain = LLMChain(llm=llm, prompt=relationship_prompt)

# ================== Processing ==================
all_results = []

for chunk in tqdm(chunks, desc="Processing chunks"):
    text = chunk.strip()
    if not text:
        continue

    # --- Extract concepts ---
    concepts_text = concept_chain.run(text=text)
    concepts_list = [c.strip() for c in concepts_text.split(",") if c.strip()]

    # --- Extract relationships ---
    if concepts_list:
        relationships_json = relationship_chain.run(
            concepts=", ".join(concepts_list),
            text=text
        )
        try:
            relationships = json.loads(relationships_json)
        except json.JSONDecodeError:
            relationships = {"error": "Invalid JSON from LLM", "raw_output": relationships_json}

        all_results.append({
            "chunk": text,
            "concepts": concepts_list,
            "relationships": relationships
        })

# ================== Save output ==================
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Saved concepts and relationships for {len(all_results)} chunks to {OUTPUT_FILE}")
