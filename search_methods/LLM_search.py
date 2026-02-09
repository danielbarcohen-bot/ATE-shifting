import os
from typing import List, Optional, Callable

import pandas as pd
from anthropic import Anthropic

from search_methods.ATE_search import ATESearch

API_KEY = ""


class Claude:
    """A clean wrapper for Anthropic's Claude models."""

    # Mapping friendly names to official API strings

    def __init__(self, api_key: Optional[str] = None):
        # Fallback to environment variable if no key is provided
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Set ANTHROPIC_API_KEY env var.")

        self.client = Anthropic(api_key=self.api_key)
        self.model_id = "sonnet-4.5"

    def ask(self, prompt: str, system: str = "You are a helpful assistant.", max_tokens: int = 1024) -> str:
        """The primary method to interact with the model."""
        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.content[0].text

            # Save for evaluation purposes
            return answer

        except Exception as e:
            return f"Error: {str(e)}"


class LLMSearch(ATESearch):
    def __init__(self, system: str, prompt: str):
        self.LLM = Claude(API_KEY)
        self.prompt = prompt
        self.system = system

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int, transformations_dict: dict[str, Callable]):

        LLM_answer = self.LLM.ask(self.prompt, self.system)
        return LLM_answer
