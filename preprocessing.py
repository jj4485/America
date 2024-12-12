from transformers import AutoProcessor, AutoModelForImageTextToText

# Load the processor and model
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

# Define your input prompt
input_text = "Describe the implications of AI in education."

# Preprocess the text input
inputs = processor(text=input_text, return_tensors="pt")

# Pass the input to the model
outputs = model.generate(**inputs)

# Decode the output tokens
response = processor.batch_decode(outputs, skip_special_tokens=True)
print(response[0])
