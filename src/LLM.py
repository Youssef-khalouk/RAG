"""
Utilities for loading a local language model
and answering questions from context.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any
import pickle
import os
from pathlib import Path


class LLM:
    """Wrap model loading, caching, context assembly, and prompting helpers."""

    model_name: str = "Qwen/Qwen3-0.6B"
    model: AutoModelForCausalLM | None = None
    tokenizer = None
    max_context_characters: int = 3000
    _content: str = (
        "You answer questions using only the provided context.\n"
        "If the answer is not in the context, say 'I don't know'.")
    llm_cache: dict = {}

    @staticmethod
    def load_cache() -> None:
        """Load the persisted LLM response cache from disk if it exists."""
        path = Path("data/processed/llm_cache.pkl")
        if path.exists():
            with open("data/processed/llm_cache.pkl", "rb") as file:
                try:
                    LLM.llm_cache = pickle.load(file)
                except Exception:
                    LLM.llm_cache = {}
                    try:
                        path.unlink()
                    except Exception:
                        pass

    @staticmethod
    def save_cache() -> None:
        """Persist the in-memory LLM response cache to disk."""
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/llm_cache.pkl", "wb") as file:
            pickle.dump(LLM.llm_cache, file)

    @staticmethod
    def _init_model() -> None:
        """Load the tokenizer and causal language model on first use."""
        # Download the tokenizer
        LLM.tokenizer = AutoTokenizer.from_pretrained(LLM.model_name)
        # Download the model
        LLM.model = AutoModelForCausalLM.from_pretrained(
            LLM.model_name,
            device_map="auto",  # Uses GPU if available
            torch_dtype="auto"  # Picks the best precision
        )
        LLM.load_cache()

    @staticmethod
    def get_context(chunks: list[Any] = []) -> str:
        """
        Assemble a context string from retrieved chunks up to
        the configured limit.
        """
        context = ""
        if chunks == []:
            return context
        if isinstance(chunks[0], dict):
            for doc in chunks:
                context += doc["text"] + "\n\n"
                if len(context) >= LLM.max_context_characters:
                    break
        elif isinstance(chunks[0], tuple):
            for doc in chunks:
                context += doc[0] + "\n\n"
                if len(context) >= LLM.max_context_characters:
                    break
        return context

    @staticmethod
    def max_tokens() -> int:
        """Return the model's maximum supported position embedding size."""
        if LLM.model is None:
            LLM._init_model()
        assert LLM.model is not None
        try:
            _max = LLM.model.config.max_position_embeddings
            return int(_max)
        except Exception:
            return 20000

    @staticmethod
    def ask(question: str, context: str = "", refresh: bool = False) -> str:
        """
        Answer a question using the provided context and cached responses.
        """
        if not LLM.model:
            LLM._init_model()
        key = (question, str(context))
        cached: str | None = LLM.llm_cache.get(key, None)
        if cached is not None:
            return cached
        message = [
            {
                "role": "system",
                "content": LLM._content
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
        assert LLM.tokenizer is not None
        text = LLM.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        assert LLM.model is not None
        inputs = LLM.tokenizer(text, return_tensors="pt").to(LLM.model.device)
        outputs = LLM.model.generate(**inputs, max_new_tokens=300)
        answer: str = LLM.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        LLM.llm_cache[key] = answer
        LLM.save_cache()
        return answer
