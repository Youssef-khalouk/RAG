from transformers import AutoTokenizer, AutoModelForCausalLM

def run_llm():
    model_name = "Qwen/Qwen3-0.6B"

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Download the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",      # Uses GPU if available
        torch_dtype="auto"      # Picks the best precision
    )

    prompt = "Explain what Python is."

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # print(f"{text}\n\n")
    # return

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200,)

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    print(answer)