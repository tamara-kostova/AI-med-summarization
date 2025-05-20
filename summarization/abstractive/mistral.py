import logging
from groq import Groq

import requests

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer
from summarization.utils import split_text_into_chunks

logger = logging.getLogger(__name__)


class MistralSummarizer(AbstractiveSummarizer):
    def __init__(
        self,
        groq_client: Groq,
        model="mistral-7b-instruct-v0.3",
        api_endpoint="http://192.168.100.66:1234/v1/chat/completions",
        groq_model="mistral-saba-24b",
        context_window=4096,
    ):
        self.groq_client = groq_client
        self.model = model
        self.api_endpoint = api_endpoint
        self.context_window = context_window
        self.groq_model = groq_model

    def generate_abstractive_summary(self, text: str) -> str:
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": "You’re an expert medical summarizer. Summarizer the following text:",
                    }
                ],
                model=self.groq_model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Mistral summarization error: {e}")
            raise

    def generate_abstractive_summary_1(self, text: str) -> str:
        try:
            headers = {"Content-Type": "application/json"}
            chunks = split_text_into_chunks(text)
            summaries = []

            for chunk in chunks:
                message = {
                    "role": "user",
                    "content": f"<|system|>You’re an expert medical summarizer.<!|user|>Summarize the following text: {chunk}",
                }
                data = {
                    "model": self.model,
                    "messages": [message],
                    "temperature": 0.5,
                    "max_tokens": 150,
                }

                response = requests.post(self.api_endpoint, headers=headers, json=data)

                if response.status_code == 200:
                    summary = response.json()["choices"][0]["message"]["content"]
                    summaries.append(summary)
                    logger.info(f"Mistral summary for chunk: {summary}")
                else:
                    logger.error(f"Chunk summarization failed: {response.text}")
                    summaries.append("(Summary unavailable for this chunk)")

            final_summary = " ".join(summaries)
            if len(summaries) > 1:
                return self.generate_abstractive_summary(final_summary)
            return final_summary

        except Exception as e:
            logger.error(f"Mistral summarization error: {e}")
            return "Summary generation failed due to internal error"
