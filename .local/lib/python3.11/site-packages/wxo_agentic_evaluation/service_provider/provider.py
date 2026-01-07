from abc import ABC, abstractmethod
from typing import List


class Provider(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def query(self, sentence: str) -> str:
        pass

    def batch_query(self, sentences: List[str]) -> List[str]:
        return [self.query(sentence) for sentence in sentences]

    @abstractmethod
    def encode(self, sentences: List[str]) -> List[list]:
        pass
