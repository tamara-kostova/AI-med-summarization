import logging

import requests

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer

logger = logging.getLogger(__name__)


class LLamaSummarizer(AbstractiveSummarizer):
    def __init__(
        self,
        model="llama-3.2-1b-instruct",
        api_endpoint="http://192.168.100.66:1234/v1/chat/completions",
    ):
        self.model = model
        self.api_endpoint = api_endpoint

    def generate_abstractive_summary(self, text: str) -> str:
        try:
            headers = {"Content-Type": "application/json"}
            message = {
                "role": "user",
                "content": f"<|system|>You’re an expert medical summarizer.<!|user|>Summarizer the following text: {text}",
            }
            data = {
                "model": self.model,
                "messages": [message],
                "temperature": 0.5,
                "max_tokens": 150,
            }
            response = requests.post(self.api_endpoint, headers=headers, json=data)

            if response.status_code == 200:
                content = response.json()
                content = content["choices"][0]
                message = content["message"]["content"]
                return message
            else:
                logger.error(f"Llama summarization error. Request failed. {response}")
                return "UNKNOWN"
        except Exception as e:
            logger.error(f"Llama summarization error: {e}")
            raise
