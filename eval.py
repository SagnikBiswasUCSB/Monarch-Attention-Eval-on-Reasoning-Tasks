import torch
import time
import lm_eval
from lm_eval import simple_evaluate
from softmax_attention import load_baseline_model
from monarch_attention import apply_monarch_monkey_patch
import pandas as pd

# Configuration
TASK = "gsm8k"
LIMIT = 100
OUTPUT_FILE = "monarch_results.txt"

def run_evaluation(model, tokenizer, model_name_tag):
    """
    Runs GSM8K evaluation on the provided model object.
    """
    print(f"\n[{model_name_tag}] Starting Evaluation on {TASK} (Limit: {LIMIT})...")
    start_time = time.time()
    
    # We use the 'hf' model type adapter from lm_eval
    # Since we are passing a loaded model, we might need to wrap it or use the arguments correctly
    # simple_evaluate supports 'model_args' but also passing the model object directly if using 'hf'
    # Actually simple_evaluate typically takes string args or initialized lm.
    # We will instantiate the HFLM wrapper manually to ensure we use our specific model instance.
    
    from lm_eval.models.huggingface import HFLM
    
    # Wrap our model
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1) 
    # Batch size 1 for accurate latency measurement per sample (approx)
    
    results = simple_evaluate(
        model=lm_obj,
        tasks=[TASK],
        limit=LIMIT,
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Extract metrics
    # GSM8K usually reports 'exact_match' (strict) or 'flexible_extract' depending on config.
    # Checking results structure
    if 'results' in results and TASK in results['results']:
        task_res = results['results'][TASK]
        # Look for accuracy metric
        acc = task_res.get('acc,none', task_res.get('exact_match,none', 0.0))
        # Or look for 'acc' key
        if 'acc' in task_res:
            acc = task_res['acc']
    else:
        acc = -1.0 # Error
        
    avg_latency = total_time / LIMIT # Rough estimate
    
    print(f"[{model_name_tag}] Finished. Acc: {acc}, Time: {total_time:.2f}s")
    
    return {
        "Model Name": model_name_tag,
        "Accuracy": acc,
        "T-Steps": LIMIT, # Total steps/samples
        "Latency per Sample": avg_latency
    }

def main():
    # 1. Load Baseline
    print(">>> Loading Baseline Model...")
    model, tokenizer = load_baseline_model()
    
    # 2. Evaluate Baseline
    baseline_stats = run_evaluation(model, tokenizer, "Phi-4-Mini-Baseline")
    
    # 3. Patch Model
    print("\n>>> Patching Model with Monarch Attention...")
    model = apply_monarch_monkey_patch(model)
    
    # 4. Evaluate Monarch
    monarch_stats = run_evaluation(model, tokenizer, "Phi-4-Mini-Monarch")
    
    # 5. Save Results
    df = pd.DataFrame([baseline_stats, monarch_stats])
    print("\n>>> Final Results:")
    print(df)
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(df.to_string(index=False))
        
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
