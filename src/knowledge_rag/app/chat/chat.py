import copy
from typing import Dict, List, Optional

from knowledge_rag.app.providers.llm_factory import get_llm_client
from knowledge_rag.app.providers.llm_interface import LLMClient


class Chat:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        system_message: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.system_message = system_message or "You are a helpful assistant."
        self.chat_history: List[Dict[str, str]] = []

    def add_message(self, message: Dict[str, str]):
        self.chat_history.append(message)

    def get_llm_client(self):
        return self.llm_client

    def set_llm_client(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def get_system_message(self):
        return self.system_message

    def set_system_message(self, system_message: str):
        self.system_message = system_message

    def get_chat_history(self):
        return self.chat_history

    def clear_chat_history(self):
        self.chat_history = []

    def send_message(self, message: Dict[str, str]):
        self._ensure_llm_client()
        self.chat_history.append(message)
        response = self.llm_client.send_message(self.system_message, self.chat_history)  # type: ignore
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def __deepcopy__(self, memo):
        print("Deepcopying chat object...")
        """Custom deepcopy that excludes the LLM client and recreates it on copy"""
        # Create a new instance without calling __init__ to avoid client creation
        new_obj = Chat.__new__(Chat)

        # Copy all attributes except llm_client
        new_obj.system_message = copy.deepcopy(self.system_message, memo)
        new_obj.chat_history = copy.deepcopy(self.chat_history, memo)

        # Create a new LLM client (will be set properly when needed)
        new_obj.llm_client = None

        return new_obj

    def _ensure_llm_client(self):
        """Ensure the LLM client is available, create if needed"""
        if self.llm_client is None:
            self.llm_client = get_llm_client("OpenAI")
