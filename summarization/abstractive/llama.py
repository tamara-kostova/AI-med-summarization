import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer

logger = logging.getLogger(__name__)


class LLamaSummarizer(AbstractiveSummarizer):
    def __init__(
        self,
        model_name="meta-llama/Llama-2-7b-chat-hf",
    ):
        import os

        os.environ["HF_HOME"] = "D:/huggingface"

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype="auto"
        )
        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )

    def generate_abstractive_summary(self, text: str) -> str:
        try:
            logger.info(f"Input text length: {len(text.split())} words")
            prompt = f"[INST] Summarize the following medical article:\n{text} [/INST]"
            print(prompt)
            if not text.strip():
                logger.warning("Empty text received for summarization")
                return ""
            if len(text.split()) < 10:
                logger.warning("Text too short for abstractive summarization")
                return text

            response = self.pipe(prompt, max_new_tokens=150, do_sample=False)
            print(response)
            return response[0]["generated_text"].split("[/INST]")[-1].strip()
        except Exception as e:
            logger.error(f"BART summarization error: {e}")
            raise
