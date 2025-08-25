import os
from typing import Dict, List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from knowledge_rag.app.providers.llm_interface import LLMClient


class AnthropicAIClient(LLMClient):
    def __init__(self, model: Optional[str] = None):
        self.model = model or "claude-3-5-sonnet-20240620"
        print(f"Using Anthropic model: {self.model}")
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def send_message(
        self,
        system_message: str,
        messages: List[Dict[str, str]],
    ) -> str:
        response = self.anthropic.messages.create(
            model=self.model,
            system=system_message,
            messages=messages,  # type: ignore
            max_tokens=4000,
        )
        return (
            response.content[0].text  # type: ignore
            if response.content[0].text  # type: ignore
            else "LLM didn't respond"
        )


if __name__ == "__main__":
    load_dotenv()
    client = AnthropicAIClient(model="claude-3-5-sonnet-20240620")
    print(
        client.send_message(
            "You are a helpful assistant.",
            [{"role": "user", "content": "What is the capital of France?"}],
        )
    )
