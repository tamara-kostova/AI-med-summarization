import logging
from groq import Groq

import requests

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer
from summarization.utils import truncate_to_6000_tokens

logger = logging.getLogger(__name__)


class LLamaSummarizer(AbstractiveSummarizer):
    def __init__(
        self,
        groq_client: Groq,
        lm_studio_model="llama-3.2-1b-instruct",
        groq_model="llama3-70b-8192",
        lm_studio_endpoint="http://192.168.100.66:1234/v1/chat/completions",
    ):
        self.groq_client = groq_client
        self.model_llm_studio_modelm_studio = lm_studio_model
        self.lm_studio_endpoint = lm_studio_endpoint
        self.groq_model = groq_model

    def generate_abstractive_summary(self, text: str) -> str:
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": truncate_to_6000_tokens(f"You’re an expert medical summarizer. Do not include any extra information in the output. Summarize the following text:{text}"),
                    }
                ],
                model=self.groq_model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Llama summarization error: {e}")
            raise

    def generate_abstractive_summary_1(self, text: str) -> str:
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
                "max_tokens": 500,
            }
            response = requests.post(self.api_endpoint, headers=headers, json=data)

            if response.status_code == 200:
                content = response.json()
                content = content["choices"][0]
                message = content["message"]["content"]
                logger.info(f"Successfully generated summary with model Llama.")
                return message
            else:
                logger.error(f"Llama summarization error. Request failed. {response}")
                return "UNKNOWN"
        except Exception as e:
            logger.error(f"Llama summarization error: {e}")
            raise
