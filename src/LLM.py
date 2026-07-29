from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM:
    model_name = "Qwen/Qwen3-0.6B"
    model = None
    tokenizer = None

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

    @staticmethod
    def ask(question: str, context: str = "") -> str:
        if not LLM.model:
            LLM._init_model()
        message = [
            {
                "role": "system",
                "content": """You answer questions using only the provided context.\nIf the answer is not in the context, say you don't know."""
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
        return LLM.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
