from abc import ABC, abstractmethod
from typing import Dict, List


class LLMClient(ABC):
    @abstractmethod
    def send_message(
        self, system_message: str, messages: List[Dict[str, str]]
    ) -> str: ...
