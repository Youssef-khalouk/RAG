from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any
import pickle
import os
from pathlib import Path


class LLM:
    model_name = "Qwen/Qwen3-0.6B"
    model = None
    tokenizer = None
    max_context_characters: int = 3000
    _content = ("You answer questions using only the provided context."
                "\nIf the answer is not in the context, say 'I don't know'.")
    llm_cache: dict = {}

    @staticmethod
    def load_cache():
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
    def save_cache():
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/llm_cache.pkl", "wb") as file:
            pickle.dump(LLM.llm_cache, file)

    @staticmethod
    def _init_model() -> None:
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
        if not LLM.model:
            LLM._init_model()

        return LLM.model.config.max_position_embeddings

    @staticmethod
    def ask(question: str, context: str = "", refresh: bool = False) -> str:
        if not LLM.model:
            LLM._init_model()
        key = (question, str(context))
        cached = LLM.llm_cache.get(key, None)
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
        text = LLM.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = LLM.tokenizer(text, return_tensors="pt").to(LLM.model.device)
        outputs = LLM.model.generate(**inputs, max_new_tokens=300)
        answer = LLM.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        LLM.llm_cache[key] = answer
        LLM.save_cache()
        return answer
