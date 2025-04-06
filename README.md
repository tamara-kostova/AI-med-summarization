# PDF Text Summarizer with Extractive and AI Abstractive Methods

This project provides a web-based tool and NLP pipeline to generate **summaries of PDF documents** using two popular techniques:

- **Extractive Summarization**: Selects the most important sentences directly from the text.
- **Abstractive Summarization**: Generates a new summary using a pre-trained transformer model, similar to how humans summarize.

It also includes a feature to **compare the performance** of both methods using **ROUGE metrics**, and recommends the better approach based on the input, along with a script to fine-tune a summarization model on the PubMed scientific papers dataset.

---

## Features

- Upload PDF files and generate summaries.
- Supports both extractive (TextRank) and abstractive (Transformer-based) summarization.
- Compare both summaries with a reference summary using ROUGE-1, ROUGE-2, and ROUGE-L.
- Automatically recommends the better method.
- Model Training Script — Train a T5-small model (or other variants) on scientific paper abstracts from PubMed.

---

## Technologies Used

- **Python** (FastAPI backend)
- **Transformers** (HuggingFace pipeline)
- **Sumy** (TextRank extractive summarizer)
- **RougeScore** for evaluation
- **HTML + JavaScript** frontend for uploading and displaying results

---

# Run application
`pip install -r requirements.txt`

`uvicorn app.main:app --reload`