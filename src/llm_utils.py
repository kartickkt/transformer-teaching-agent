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

def call_llm(
    prompt: str,
    temperature: float = 0.0,   # ignored if do_sample=False
    max_new_tokens: int = 256,
    do_sample: bool = False      # default: greedy decoding
) -> str:
    """
    Calls the instruction-tuned Mistral model and returns the text output.

    Parameters:
    - prompt: text input to the model
    - temperature: sampling temperature (used only if do_sample=True)
    - max_new_tokens: max tokens to generate
    - do_sample: whether to use sampling (True) or greedy decoding (False)
    """
    # Safety check for temperature
    if do_sample and temperature <= 0.0:
        raise ValueError("If do_sample=True, temperature must be > 0.0")

    outputs = llm_pipeline(
        prompt,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        max_new_tokens=max_new_tokens
    )

    # Remove the prompt prefix from output
    return outputs[0]["generated_text"][len(prompt):].strip()
