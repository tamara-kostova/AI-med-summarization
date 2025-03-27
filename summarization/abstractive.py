from transformers import pipeline

summarizer = pipeline("summarization", model="t5-small")

def generate_abstractive_summary(text, max_length=150):
    return summarizer(text, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
