from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Your prompt
prompt = "What is the capital of France?"

# Tokenize prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate model output
outputs = model.generate(
    **inputs,
    max_length=50,
    temperature=0.7,
    top_p=0.9
)

# Decode the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model response:", response)
