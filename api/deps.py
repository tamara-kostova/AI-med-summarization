from fastapi import Request

from summarization.evaluator import Evaluator
from summarization.summarizer import Summarizer


def get_summarizer(request: Request) -> Summarizer:
    return request.app.state.summarizer  # type: ignore


def get_evaluator(request: Request) -> Evaluator:
    return request.app.state.evaluator  # type: ignore
