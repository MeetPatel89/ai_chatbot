import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from knowledge_rag.app.providers.llm_interface import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model or "gpt-4o-mini"
        print(f"OpenAI client initialized with model: {self.model}")
        self.client = OpenAI(api_key=self.api_key)

    def send_message(self, system_message: str, messages: List[Dict[str, str]]) -> str:
        messages = [
            {"role": "system", "content": system_message},
            *messages,
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
        )
        return (
            response.choices[0].message.content
            if response.choices[0].message.content
            else "LLM didn't respond"
        )


if __name__ == "__main__":
    load_dotenv()
    client = OpenAIClient(model="gpt-4o-mini")

    # Example with message history
    message_history = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    system_message = "You are a helpful assistant. You give detailed answers and are frank when you don't know the answer."

    print(client.send_message(system_message, message_history))
