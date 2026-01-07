import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_NAME = "microsoft/Phi-4-mini-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_baseline_model():
    """
    Loads the baseline Phi-4-Mini model in its standard configuration.
    """
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        dtype=torch.float32,
        device_map="auto",
        trust_remote_code=False
    )
    
    return model, tokenizer

def run_baseline_inference(prompt):
    """
    Runs a simple inference pass to verify baseline functionality.
    """
    model, tokenizer = load_baseline_model()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Baseline Output ---")
    print(response)
    print("-----------------------")
    
    return response

if __name__ == "__main__":
    # Test Prompt
    PROMPT = "Explain the concept of attention in neural networks broadly."
    run_baseline_inference(PROMPT)
