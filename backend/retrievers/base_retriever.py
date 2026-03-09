from abc import ABC, abstractmethod
from typing import Any, List


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Any]:
        raise NotImplementedError