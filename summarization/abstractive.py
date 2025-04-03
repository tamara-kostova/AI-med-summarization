from transformers import AutoTokenizer, pipeline
import os

def get_latest_model(base_dir="./model_checkpoints"):
    """
    Find the most recently trained model
    """
    if not os.path.exists(base_dir):
        return "t5-small"
   
    model_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
   
    return max(model_dirs, key=os.path.getmtime) if model_dirs else "t5-small"

def generate_abstractive_summary(text, max_length=150):
    try:
        # Use get_latest_model to find the most recent model
        latest_model = get_latest_model()
        print(latest_model)
        
        # Create tokenizer using the latest model
        tokenizer = AutoTokenizer.from_pretrained(latest_model, use_fast=False)
        
        # Create pipeline with the latest model and its tokenizer
        summarizer = pipeline(
            "summarization", 
            model=latest_model,
            tokenizer=tokenizer
        )
        
        return summarizer(text, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
    except Exception as e:
        print(f"Error in summarization: {e}")
        return text  # Fallback to original text if summarization fails