import json
import os
from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore

# Import the correct class for running Hugging Face models locally
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt

# --- Load the JSON data from the file ---
# The logic to load your JSON file is correct, with a slight simplification
file_path = os.path.join("outputs", "phase2_concepts.json")
try:
    with open(file_path, 'r') as f:
        concepts_json = json.load(f)
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    exit()

# --- Setup the LLM for Local Inference ---
print("Setting up the Hugging Face LLM...")
# Define the prompt templates required for the Hugging Face model
system_prompt = "You are a helpful assistant. Your task is to extract relationships between concepts from the text."
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

# Instantiate the HuggingFaceLLM class
# This will automatically download and load the model onto your GPU
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.2, "do_sample": True},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto"  # This is crucial for GPU usage
)
print("LLM setup complete!")

# --- Define your documents from the chunks ---
documents = [
    Document(text=d["chunk"], metadata={"chunk_id": d["chunk_id"]})
    for d in concepts_json
]

# --- Build the Graph ---
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

entity_types = ["Concept"]
relationship_types = ["IS_A", "REPLACES", "BENEFITS_FROM", "INVOLVES", "ACHIEVES"]

print("Building the Knowledge Graph... This may take some time.")
index = KnowledgeGraphIndex.from_documents(
    documents,
    llm=llm,
    storage_context=storage_context,
    kg_extractors=KnowledgeGraphIndex.get_kg_extractors(
        kg_extractors_list=["simple", "schema_enforced"],
        llm=llm,
        kg_rel_types=relationship_types,
    ),
)
print("Knowledge Graph built successfully! 🎉")

# --- Save the Output ---
persist_dir = "./attention_kg_data"

if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

print(f"Persisting the knowledge graph to {persist_dir}...")
index.storage_context.persist(persist_dir=persist_dir)
print("Done! The graph has been saved.")