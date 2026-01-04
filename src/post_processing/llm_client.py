"""
Wrapper for OpenAI API calls
"""
from __future__ import annotations
import os
from typing import Optional
from openai import OpenAI

from src.pre_processing.env_setup import ensure_openai_api_key


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-5.2"):
        ensure_openai_api_key()
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, timeout=600)
    
    def call(self, system_message: str, user_message: str, temperature: float = 0) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content.strip()
