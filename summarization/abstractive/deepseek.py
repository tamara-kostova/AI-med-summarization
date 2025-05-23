import logging

from groq import Groq
import requests

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer

logger = logging.getLogger(__name__)


class DeepSeekSummarizer(AbstractiveSummarizer):
    def __init__(
        self,
        groq_client: Groq,
        lm_studio_model="deepseek-r1-distill-llama-8b",
        groq_model="deepseek-r1-distill-llama-70b",
        lm_studio_endpoint="http://192.168.100.66:1234/v1/chat/completions",
    ):
        self.groq_client = groq_client
        self.lm_studio_model = lm_studio_model
        self.lm_studio_endpoint = lm_studio_endpoint
        self.groq_model = groq_model

    def generate_abstractive_summary(self, text: str) -> str:
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"You’re an expert medical summarizer. Summarizer the following text: {text}",
                    }
                ],
                model=self.groq_model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Deepseek summarization error: {e}")
            raise

    def generate_abstractive_summary_1(self, text: str) -> str:
        try:
            headers = {"Content-Type": "application/json"}
            messages = [
                {"role": "system", "content": "You're an expert medical summarizer."},
                {"role": "user", "content": f"Summarize this medical text: {text}"},
            ]
            data = {
                "model": self.lm_studio_model,
                "messages": [messages],
                "temperature": 0.5,
                "max_tokens": 150,
            }
            response = requests.post(
                self.lm_studio_endpoint, headers=headers, json=data
            )

            if response.status_code == 200:
                content = response.json()
                if "choices" in content and len(content["choices"]) > 0:
                    logger.info(f"Successfully generated summary with model DeepSeek.")
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
