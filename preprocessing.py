from transformers import LlamaTokenizer, LlamaForCausalLM
import os
# Load the tokenizer and model from your local directory
current_path = os.getcwd()
tokenizer = LlamaTokenizer.from_pretrained(current_path, "hf-model-llama-7b")
model = LlamaForCausalLM.from_pretrained(current_path, "hf-model-llama-7b")

# Prepare input
input_text = "Hello World"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
