# Medical PDF Text Summarizer with Extractive, Abstractive, and Hybrid AI Methods

This project provides a web-based tool and NLP pipeline to generate summaries of medical PDF documents using state-of-the-art extractive, abstractive, and hybrid summarization techniques. It is designed for medical and scientific literature, supporting robust comparison and evaluation.

## Features

- PDF Upload & Summarization: Upload medical/scientific PDFs and generate summaries instantly.

- Extractive Summarization: Selects important sentences using methods like TextRank (with MMR diversity), LexRank, LSA, BERT, and SummaRuNNer.
  
- Abstractive Summarization: Generates new summaries using advanced transformer models (Llama, DeepSeek, Mistral, BART, T5, ProphetNet).

- Hybrid Summarization: Combines extractive (TextRank+MMR) and abstractive (LLama or other LLM) for improved coverage and readability.

- Automatic Evaluation & Recommendation: Compares generated summaries against reference summaries using ROUGE, BLEU, and BERTScore, and recommends the best approach.

- Model Training Script: Fine-tune abstractive models from HuggingFace on large-scale scientific datasets (e.g., PubMed).

- User-friendly Web Interface: Simple upload, summarization, and results display.

---

## Technologies Used

- **Python** (FastAPI backend)
- **Transformers** (HuggingFace pipeline)
- **Sumy** (TextRank extractive summarizer)
- **Scikit-learn** (TF-IDF and MMR for diversity)
- **RougeScore, BLEU, BERTScore** (for evaluation)
- **HTML + JavaScript** frontend for uploading and displaying results

---

# Run application
`pip install -r requirements.txt`

`uvicorn app.main:app --reload`
