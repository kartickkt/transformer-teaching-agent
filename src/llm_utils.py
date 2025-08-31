from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # Instruction-tuned model

print("Loading model. This may take a few minutes...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model with automatic device placement
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",         # automatically put layers on GPU if available
    torch_dtype=torch.float16, # use FP16 for faster inference on compatible GPUs
    low_cpu_mem_usage=True
)

# Create pipeline WITHOUT specifying device
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print("Model loaded successfully!")

def call_llm(prompt: str, temperature: float = 0.7, max_new_tokens: int = 256) -> str:
    """
    Calls the instruction-tuned Mistral model and returns the text output.
    """
    outputs = llm_pipeline(
        prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=True
    )
    # Remove the prompt prefix from output
    return outputs[0]["generated_text"][len(prompt):].strip()
