import json
import os
import re
import time
from typing import List, Optional, Callable

import pandas as pd
from anthropic import Anthropic

from search_methods.ATE_search import ATESearch
from secret import API_KEY_SONNET
from utils import calculate_ate_linear_regression_lstsq, apply_data_preparations_seq, list_seq_to_tuple_seq


class Claude:
    """A clean wrapper for Anthropic's Claude models."""

    # Mapping friendly names to official API strings

    def __init__(self, api_key: Optional[str] = None):
        # Fallback to environment variable if no key is provided
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Set ANTHROPIC_API_KEY env var.")

        self.client = Anthropic(api_key=self.api_key)
        self.model_id = "claude-sonnet-4-5-20250929"  # "claude-haiku-4-5"#"claude-sonnet-4-5-20250929"

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
        self.LLM = Claude(API_KEY_SONNET)
        self.prompt = prompt
        self.system = system

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int, transformations_dict: dict[str, Callable]):
        start_time = time.time()
        LLM_answer = self.LLM.ask(self.prompt, self.system)
        print(LLM_answer)
        print("took: ", time.time() - start_time)
        sequence = json.loads(re.search(r'\[\s*{.*?}\s*\]', LLM_answer, re.DOTALL).group())
        df_transformed = apply_data_preparations_seq(df.copy(), list_seq_to_tuple_seq(sequence), transformations_dict)
        print(f"ATE IS: {calculate_ate_linear_regression_lstsq(df_transformed, 'treatment', 'outcome', common_causes)}")
        return LLM_answer

# if __name__ == '__main__':
#     claude = Claude(API_KEY_SONNET)
#     print(claude.ask("How old is you?", system="You are a helpful assistant."))
