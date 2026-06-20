from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    @abstractmethod
    def build(self, chunks: list[dict]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5, source: str | None = None) -> list[dict]:
        pass