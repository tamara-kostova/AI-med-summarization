import logging

import pymupdf
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tokenizer = tiktoken.get_encoding("cl100k_base")

def split_text_into_chunks(text, max_words=1000):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i : i + max_words])


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text content from PDF bytes"""
    try:
        doc = pymupdf.open(stream=file_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        if not text.strip():
            logger.warning("No text extracted from PDF")
            return "No readable text found in the PDF document."

        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def truncate_to_6000_tokens(text: str) -> str:
    tokens = tokenizer.encode(text)
    truncated_tokens = tokens[:6000]
    logger.info(f"Truncated to: {len(truncated_tokens)}")
    return tokenizer.decode(truncated_tokens)