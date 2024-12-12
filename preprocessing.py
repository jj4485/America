from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-1B")