from abc import ABC, abstractmethod


class ExtractiveSummarizer(ABC):
    @abstractmethod
    def generate_extractive_summary(self):
        pass
