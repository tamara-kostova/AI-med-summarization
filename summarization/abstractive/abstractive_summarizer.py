from abc import ABC, abstractmethod


class AbstractiveSummarizer(ABC):
    @abstractmethod
    def generate_abstractive_summary(self):
        pass
