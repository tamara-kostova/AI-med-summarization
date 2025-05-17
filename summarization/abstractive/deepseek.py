import logging

import requests

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer

logger = logging.getLogger(__name__)


class DeepSeekSummarizer(AbstractiveSummarizer):
    def __init__(
        self,
        model="deepseek-r1-distill-llama-8b",
        api_endpoint="http://192.168.100.66:1234/v1/chat/completions",
    ):
        self.model = model
        self.api_endpoint = api_endpoint

    def generate_abstractive_summary(self, text: str) -> str:
        try:
            headers = {"Content-Type": "application/json"}
            messages = [
                {"role": "system", "content": "You're an expert medical summarizer."},
                {"role": "user", "content": f"Summarize this medical text: {text}"},
            ]
            data = {
                "model": self.model,
                "messages": [messages],
                "temperature": 0.5,
                "max_tokens": 150,
            }
            response = requests.post(self.api_endpoint, headers=headers, json=data)

            if response.status_code == 200:
                content = response.json()
                if "choices" in content and len(content["choices"]) > 0:
                    return content["choices"][0]["message"]["content"].strip()
                else:
                    logger.error("Empty response from DeepSeek API")
                    return ""
            else:
                logger.error(
                    f"DeepSeek summarization error. Request failed. {response.json()}"
                )
                return "DeepSeek Summary generation failed :()"
        except Exception as e:
            logger.error(f"DeepSeek summarization error: {e}")
            raise
